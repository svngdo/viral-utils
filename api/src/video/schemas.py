from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from src.config import settings
from src.inpainting.schemas import InpaintEngine
from src.llm.constants import LLM_MODELS
from src.llm.schemas import LLMModel
from src.ocr.schemas import OcrEngine


class VideoEngine(StrEnum):
    PYAV = "pyav"
    FFMPEG = "ffmpeg"
    OPENCV = "opencv"


class VideoCodec(StrEnum):
    H264_VIDEOTOOLBOX = "h264_videotoolbox"  # mac hw h264
    HEVC_VIDEOTOOLBOX = "hevc_videotoolbox"  # mac hw hevc (default)
    LIBX264 = "libx264"  # sw h264, cross-platform
    LIBX265 = "libx265"  # sw hevc, cross-platform


class VideoConfig(BaseModel):
    codec: VideoCodec
    quality: int


class ProcessVideosRequest(BaseModel):
    # I/O
    in_dir: str = str(settings.raw_dir)
    out_dir: str = str(settings.process_dir)

    # Video
    video_engine: VideoEngine = VideoEngine.FFMPEG
    video_codec: VideoCodec = VideoCodec.HEVC_VIDEOTOOLBOX
    video_quality: int = Field(default=80, ge=0, le=100)
    # OCR
    ocr_engine: OcrEngine = OcrEngine.OCRMAC
    ocr_sample_interval: int = 3
    ocr_delay: float = 0.3
    ocr_chinese_only: bool = True
    # Subtitle
    sub_time_gap_tolerance: float = 0.5
    sub_text_similarity_threshold: float = 0.5
    sub_box_iou_threshold: float = 0.5
    sub_frame_padding: int = 3
    # Inpaint
    inpaint_engine: InpaintEngine = InpaintEngine.OPENCV
    inpaint_conf_threshold: float = 0.3
    inpaint_scale: float = 0.85
    inpaint_expand: int = 10
    inpaint_radius: int = 5
    inpaint_delay: float = 0.05

    # LLM
    llm_models: list[LLMModel] = Field(
        default_factory=lambda: [LLMModel(**m) for m in LLM_MODELS]
    )

    @field_validator("in_dir")
    @classmethod
    def input_required(cls, v: str) -> str:
        path = Path(v).expanduser()
        if not path.exists():
            raise ValueError(f"input directory does not exist: {path}")
        return str(path)

    @field_validator("out_dir")
    @classmethod
    def normalize_out_dir(cls, v: str) -> str:
        return str(Path(v).expanduser())

    @field_validator("llm_models")
    @classmethod
    def llm_models_required(cls, v):
        if not v:
            raise ValueError("at least one LLM model is required")
        return v
