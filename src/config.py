import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_NAME = "viral-utils"
_HOME = Path.home()


@dataclass
class Config:
    # --- Environment ---
    log_level: str
    gemini_api_key: str

    groq_api_key: str
    groq_base_url: str

    openrouter_api_key: str
    openrouter_base_url: str

    cerebras_api_key: str
    cerebras_base_url: str

    # --- Paths ---
    cache_dir: Path
    sources_dir: Path
    raw_dir: Path
    processed_dir: Path
    exports_dir: Path
    archives_dir: Path
    archives_raw_dir: Path
    downloads_dir: Path


def load_config() -> Config:
    load_dotenv()
    return Config(
        log_level=os.getenv("LOG_LEVEL", "debug"),
        # Gemini
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        # Groq
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_base_url=os.getenv("GROQ_BASE_URL", ""),
        # Openrouter
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", ""),
        # Cerebras
        cerebras_base_url=os.getenv("CEREBRAS_BASE_URL", ""),
        cerebras_api_key=os.getenv("CEREBRAS_API_KEY", ""),
        # Dir
        cache_dir=Path(os.getenv("CACHE_DIR", str(_HOME / f".cache/{PROJECT_NAME}"))),
        sources_dir=Path(
            os.getenv("SOURCES_DIR", str(_HOME / "Desktop/douyin/0_sources"))
        ),
        raw_dir=Path(os.getenv("RAW_DIR", str(_HOME / "Desktop/douyin/1_raw"))),
        processed_dir=Path(
            os.getenv("PROCESSED_DIR", str(_HOME / "Desktop/douyin/2_processed"))
        ),
        exports_dir=Path(
            os.getenv("EXPORTS_DIR", str(_HOME / "Desktop/douyin/3_exports"))
        ),
        archives_dir=Path(
            os.getenv("ARCHIVES_DIR", str(_HOME / "Desktop/douyin/4_archives"))
        ),
        archives_raw_dir=Path(
            os.getenv("ARCHIVES_RAW_DIR", str(_HOME / "Desktop/douyin/4_archives_raw"))
        ),
        downloads_dir=Path(
            os.getenv("DOWNLOADS_DIR", str(_HOME / "Desktop/douyin/5_downloads"))
        ),
    )


settings = load_config()
