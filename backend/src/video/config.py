import json
from pathlib import Path

from pydantic_settings import BaseSettings

from src.video.schemas import VideoConfigUpdate

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
    # --- Dir ---
    base_dir: Path = Path("~/Desktop/douyin").expanduser()

    @property
    def sources_dir(self) -> Path:
        return self.base_dir / "0_sources"

    @property
    def raw_dir(self) -> Path:
        return self.base_dir / "1_raw"

    @property
    def processed_dir(self) -> Path:
        return self.base_dir / "2_processed"

    @property
    def exports_dir(self) -> Path:
        return self.base_dir / "3_exports"

    @property
    def archives_dir(self) -> Path:
        return self.base_dir / "4_archives"

    @property
    def downloads_dir(self) -> Path:
        return self.base_dir / "5_downloads"


def load_video_config() -> VideoConfig:
    if CONFIG_PATH.exists():
        data = json.loads(CONFIG_PATH.read_text())
        return VideoConfig(**data)  # override defaults
    return VideoConfig()


def save_video_config(video_config: VideoConfig) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(video_config.model_dump_json(indent=2))


def update_video_config(video_config: VideoConfigUpdate) -> VideoConfig:
    current = load_video_config()
    patch = video_config.model_dump(exclude_unset=True)
    updated = current.model_copy(update=patch)
    save_video_config(updated)
    return updated


def reset_config() -> VideoConfig:
    CONFIG_PATH.unlink(missing_ok=True)
    return VideoConfig()


video_config = load_video_config()
