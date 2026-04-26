from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from src.video.constants import ALLOWED_EXTENSIONS


class VideoMetadata(BaseModel):
    width: int
    height: int
    fps: float
    total_frames: int
    color_range: str
    color_space: str
    color_primaries: str
    color_transfer: str
    pix_fmt: str
    duration: float


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class Subtitle(BaseModel):
    text: str
    conf: float
    bbox: BoundingBox
    start: float
    end: float


class VideoConfigUpdate(BaseModel):
    # --- OCR ---
    ocr_engine_name: str | None = None
    ocr_sample_interval: int | None = None
    ocr_chinese_only: bool | None = None
    ocr_delay: float | None = None
    # --- Subtitle ---
    sub_time_gap_tolerance: float | None = None
    sub_text_similarity_threshold: float | None = None
    sub_box_iou_threshold: float | None = None
    sub_frame_padding: int | None = None
    # --- Translate ---
    translate_conf_threshold: float | None = None
    # --- Inpaint ---
    inpaint_conf_threshold: float | None = None
    inpaint_scale: float | None = None
    inpaint_expand: int | None = None
    inpaint_radius: int | None = None
    inpaint_delay: float | None = None
    # --- Output ---
    output_dir: Path | None = None


class InpaintRequest(BaseModel):
    input_path: str
    output_path: str

    @field_validator("input_path")
    @classmethod
    def validate_video_path(cls, input_path: str) -> Path:
        path = Path(input_path)
        if not path.exists() or not path.is_file():
            raise ValueError("File not found")
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValueError("Unsupported format")
        return path


class InpaintResponse(BaseModel):
    output_path: str
    srt_path: str


class InpaintStatus(StrEnum):
    OK = "OK"
    ERROR = "ERROR"


class InpaintResult(BaseModel):
    file: str
    status: InpaintStatus
    error: str | None = None


class InpaintAllResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: list[InpaintResult]


class OcrResult(BaseModel):
    video_path: str
    subtitles: list[Subtitle] = Field(default_factory=list)
