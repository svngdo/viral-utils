import json
from pathlib import Path

from pydantic_settings import BaseSettings

from src.config import settings

CONFIG_PATH = Path.home() / ".viral-utils" / "video_config.json"


class VideoConfig(BaseSettings):
    # --- OCR ---
    ocr_engine_name: str = "ocrmac"
    ocr_sample_interval: int = 3
    ocr_chinese_only: bool = True
    ocr_delay: float = 0.36  # seconds

    # --- Subtitle ---
    sub_time_gap_tolerance: float = 0.6
    sub_text_similarity_threshold: float = 0.6
    sub_box_iou_threshold: float = 0.6
    sub_frame_padding: int = 3

    # --- Translate ---
    translate_conf_threshold: float = 0.5

    # --- Inpaint ---
    inpaint_conf_threshold: float = 0.3
    inpaint_scale: float = 0.6
    inpaint_expand: int = 6
    inpaint_radius: int = 6
    inpaint_delay: float = 0.06

    # --- Output ---
    output_dir: Path = settings.processed_dir


def load_video_config() -> VideoConfig:
    if CONFIG_PATH.exists():
        data = json.loads(CONFIG_PATH.read_text())
        return VideoConfig(**data)  # override defaults
    return VideoConfig()


def save_config(config: VideoConfig) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(config.model_dump_json(indent=2))


def reset_config() -> None:
    CONFIG_PATH.unlink(missing_ok=True)


video_config = load_video_config()
