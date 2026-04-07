import os

from google import genai
from google.genai import types
from openai import OpenAI

from src.config import settings
from src.logging import get_logger

logger = get_logger(__name__)


# Gemini models - tried in order, same key
# Free tier via Google AI Studio API key
GEMINI_MODELS = [
    "gemini-3-flash-preview",  # 20 req/day, 5 rpm
    "gemini-3.1-flash-lite-preview",  # 500 req/day, 15 rpm
    "gemini-2.5-flash",  # 20 req/day, 5 rpm
    "gemini-2.5-flash-lite",  # 20 req/day, 10 rpm
]

# (api_key_env, base_url_constant, model_id)
OPENAI_COMPAT_MODELS = [
    # --- OpenRouter free models ---
    # 50 req/day free (1000/day with $10 lifetime topup), 20 rpm
    # Best quality
    ("OPENROUTER_API_KEY", settings.openrouter_base_url, "deepseek/deepseek-r1:free"),
    (
        "OPENROUTER_API_KEY",
        settings.openrouter_base_url,
        "deepseek/deepseek-chat-v3.1:free",
    ),
    (
        "OPENROUTER_API_KEY",
        settings.openrouter_base_url,
        "meta-llama/llama-4-maverick:free",
    ),
    ("OPENROUTER_API_KEY", settings.openrouter_base_url, "qwen/qwen3-235b-a22b:free"),
    ("OPENROUTER_API_KEY", settings.openrouter_base_url, "openai/gpt-oss-120b:free"),
    ("OPENROUTER_API_KEY", settings.openrouter_base_url, "openai/gpt-oss-20b:free"),
    ("OPENROUTER_API_KEY", settings.openrouter_base_url, "qwen/qwen3-coder:free"),
    (
        "OPENROUTER_API_KEY",
        settings.openrouter_base_url,
        "meta-llama/llama-3.3-70b-instruct:free",
    ),
    (
        "OPENROUTER_API_KEY",
        settings.openrouter_base_url,
        "nousresearch/hermes-3-llama-3.1-405b:free",
    ),
    (
        "OPENROUTER_API_KEY",
        settings.openrouter_base_url,
        "mistralai/mistral-small-3.1-24b-instruct:free",
    ),
    ("OPENROUTER_API_KEY", settings.openrouter_base_url, "google/gemma-3-27b-it:free"),
    ("OPENROUTER_API_KEY", settings.openrouter_base_url, "google/gemma-3-12b-it:free"),
    (
        "OPENROUTER_API_KEY",
        settings.openrouter_base_url,
        "nvidia/nemotron-3-nano-30b-a3b:free",
    ),
    # --- Groq free tier ---
    # All models accessible free, rate-limited only, no credit card
    (
        "GROQ_API_KEY",
        settings.groq_base_url,
        "openai/gpt-oss-120b",
    ),  # 1000 req/day, 8k tpm
    (
        "GROQ_API_KEY",
        settings.groq_base_url,
        "openai/gpt-oss-20b",
    ),  # 1000 req/day, 8k tpm
    (
        "GROQ_API_KEY",
        settings.groq_base_url,
        "moonshotai/kimi-k2-instruct-0905",
    ),  # 1000 req/day, 10k tpm
    (
        "GROQ_API_KEY",
        settings.groq_base_url,
        "llama-3.3-70b-versatile",
    ),  # 1000 req/day, 12k tpm
    (
        "GROQ_API_KEY",
        settings.groq_base_url,
        "meta-llama/llama-4-scout-17b-16e-instruct",
    ),  # 1000 req/day, 30k tpm
    (
        "GROQ_API_KEY",
        settings.groq_base_url,
        "meta-llama/llama-4-maverick-17b-128e-instruct",
    ),  # 1000 req/day
    ("GROQ_API_KEY", settings.groq_base_url, "qwen/qwen3-32b"),  # 1000 req/day, 6k tpm
    ("GROQ_API_KEY", settings.groq_base_url, "llama-3.1-8b-instant"),
    # --- Cerebras free tier ---
    # 14400 req/day, 1M tokens/day — very generous
    ("CEREBRAS_API_KEY", settings.cerebras_base_url, "gpt-oss-120b"),
    ("CEREBRAS_API_KEY", settings.cerebras_base_url, "qwen3-235b-a22b"),
    ("CEREBRAS_API_KEY", settings.cerebras_base_url, "llama-3.3-70b"),
    ("CEREBRAS_API_KEY", settings.cerebras_base_url, "qwen3-32b"),
    ("CEREBRAS_API_KEY", settings.cerebras_base_url, "llama3.1-8b"),
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


def _call_openai(
    api_key: str, base_url: str, model: str, system_prompt: str, content: str
) -> str:
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
