from src.config import settings
from src.llm.constants import (
    CEREBRAS_BASE_URL,
    GROQ_BASE_URL,
    OPENROUTER_BASE_URL,
)
from src.llm.provider import GeminiProvider, LLMProvider, OpenAICompatProvider


def build_free_providers() -> list[LLMProvider]:
    gemini_models = [
        GeminiProvider(model_id=model_id)
        for model_id in [
            "gemini-3.1-flash-lite-preview",
            "gemini-3-flash-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
        ]
    ]
    openrouter_models = [
        OpenAICompatProvider(
            model_id=model_id,
            api_key=settings.openrouter_api_key,
            base_url=OPENROUTER_BASE_URL,
        )
        for model_id in [
            "deepseek/deepseek-r1:free",
            "deepseek/deepseek-chat-v3.1:free",
            "meta-llama/llama-4-maverick:free",
            "qwen/qwen3-235b-a22b:free",
            "openai/gpt-oss-120b:free",
            "openai/gpt-oss-20b:free",
            "qwen/qwen3-coder:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "nousresearch/hermes-3-llama-3.1-405b:free",
            "mistralai/mistral-small-3.1-24b-instruct:free",
            "nvidia/nemotron-3-nano-30b-a3b:free",
        ]
    ]
    groq_models = [
        OpenAICompatProvider(
            model_id=model_id,
            api_key=settings.groq_api_key,
            base_url=GROQ_BASE_URL,
        )
        for model_id in [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "moonshotai/kimi-k2-instruct-0905",
            "llama-3.3-70b-versatile",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "qwen/qwen3-32bllama-3.1-8b-instant",
        ]
    ]

    cerebras_models = [
        OpenAICompatProvider(
            model_id=model_id,
            api_key=settings.cerebras_api_key,
            base_url=CEREBRAS_BASE_URL,
        )
        for model_id in [
            "gpt-oss-120b",
            "qwen3-235b-a22b",
            "llama-3.3-70b",
            "qwen3-32b",
            "llama3.1-8b",
        ]
    ]

    return [
        *gemini_models,
        *openrouter_models,
        *groq_models,
        *cerebras_models,
    ]
