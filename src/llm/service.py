import os

from google import genai
from google.genai import types
from openai import OpenAI

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


# Gemini models - tried in order, same key
# gemini-1.5 and gemini-2.0 are shut down / shutting down June 1 2026
GEMINI_MODELS = [
    "gemini-3.1-pro-preview",  # most capable, paid
    "gemini-3-flash-preview",  # free tier available
    "gemini-3.1-flash-lite-preview",  # free tier, fastest/cheapest
    "gemini-2.5-flash",  # stable fallback
    "gemini-2.5-flash-lite",  # stable cheap fallback
]

# (api_key_env, base_url_constant, model_id)
OPENAI_COMPAT_MODELS = [
    # OpenRouter free models
    ("OPENROUTER_API_KEY", settings.openrouter_base_urL, "deepseek/deepseek-r1:free"),
    ("OPENROUTER_API_KEY", settings.openrouter_base_urL, "deepseek/deepseek-chat-v3-1:free"),
    ("OPENROUTER_API_KEY", settings.openrouter_base_urL, "meta-llama/llama-4-maverick:free"),
    ("OPENROUTER_API_KEY", settings.openrouter_base_urL, "qwen/qwen3-235b-a22b:free"),
    # Groq free tier (rate-limited but no cost)
    ("GROQ_API_KEY", settings.groq_base_url, "openai/gpt-oss-120b"),
    ("GROQ_API_KEY", settings.groq_base_url, "llama-3.1-8b-instant"),
]


def _call_gemini(model: str, system_prompt: str, content: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=content,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3,
        ),
    )
    return response.text.strip()


def _call_openai(api_key: str, base_url: str, model: str, system_prompt: str, content: str) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


class LLMClient:
    def complete(self, system_prompt: str, content: str) -> str:
        errors = []

        for model in GEMINI_MODELS:
            try:
                result = _call_gemini(model, system_prompt, content)
                logger.info("LLM response via %s", model)
                return result
            except Exception as e:
                code = getattr(e, "code", None) or getattr(e, "status_code", None)
                logger.debug("Skipping %s (%s)", model, code or type(e).__name__)
                errors.append(f"{model}: {code or e}")

        for key_env, base_url, model in OPENAI_COMPAT_MODELS:
            api_key = os.getenv(key_env)
            if not api_key:
                continue
            try:
                result = _call_openai(api_key, base_url, model, system_prompt, content)
                logger.info("LLM response via %s", model)
                return result
            except Exception as e:
                code = getattr(e, "code", None) or getattr(e, "status_code", None)
                logger.debug("Skipping %s (%s)", model, code or type(e).__name__)
                errors.append(f"{model}: {code or e}")

        raise RuntimeError("All models failed:\n" + "\n".join(errors))
