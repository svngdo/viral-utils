from pathlib import Path

from fastapi import HTTPException

from src.video.config import VideoConfig, video_config
from src.video.constants import ALLOWED_EXTENSIONS


def validate_video_path(video_path: str) -> Path:
    path = Path(video_path)
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "File not found")
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Unsupported format")
    return path


def validate_video_dir(video_dir: str) -> Path:
    path = Path(video_dir)
    if not path.exists() or not path.is_dir():
        raise HTTPException(404, "Folder not found")
    return path


def get_video_config() -> VideoConfig:
    return video_config
