import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Config:
    # --- Environment ---
    log_level: str
    # --- Api keys ---
    gemini_api_key: str
    groq_api_key: str
    openrouter_api_key: str
    cerebras_api_key: str


def load_config() -> Config:
    load_dotenv()
    return Config(
        log_level=os.getenv("LOG_LEVEL", "debug"),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        cerebras_api_key=os.getenv("CEREBRAS_API_KEY", ""),
    )


settings = load_config()
