from pydantic import BaseModel, Field


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


class InpaintRequest(BaseModel):
    video_path: str


class InpaintAllRequest(BaseModel):
    video_dir: str


class OcrResult(BaseModel):
    video_path: str
    subtitles: list[Subtitle] = Field(default_factory=list)
