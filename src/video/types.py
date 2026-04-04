from dataclasses import dataclass


@dataclass
class VideoMetadata:
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


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class Subtitle:
    text: str
    conf: float
    bbox: BoundingBox
    start: float
    end: float
