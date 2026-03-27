import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class Config:
    gemini_api_key: str

    log_level: str = "debug"

    # Path
    data_dir: Path = PROJECT_ROOT / "data"
    log_dir: Path = PROJECT_ROOT / "log"
    out_dir: Path = PROJECT_ROOT / "out"
    tmp_dir: Path = PROJECT_ROOT / "tmp"


def load_config() -> Config:
    load_dotenv()
    config = Config(gemini_api_key=os.getenv("GEMINI_API_KEY", ""))

    for name, value in vars(config).items():
        if "_dir" in name:
            Path(value).mkdir(parents=True, exist_ok=True)

    return config


settings = load_config()
