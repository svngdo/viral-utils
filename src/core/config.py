import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Config:
    log_level: str
    gemini_api_key: str


def load_config() -> Config:
    load_dotenv()
    config = Config(
        log_level=os.getenv("LOG_LEVEL", "debug"),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
    )
    return config


settings = load_config()
