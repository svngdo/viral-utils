import time
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import numpy as np
from ocrmac import ocrmac
from ocrmac.ocrmac import OCR

from src.logging import get_logger
from src.video.config import VideoConfig
from src.video.ffmpeg import iter_frames
from src.video.types import BoundingBox, Subtitle, VideoMetadata

logger = get_logger(__name__)


class OcrEngine(Protocol):
    def detect(self, frame: np.ndarray, metadata: VideoMetadata) -> list[Subtitle]:
        """Detect text in a single frame and return a list of Subtitle."""
        ...


class Ocrmac(OcrEngine):
    def __init__(self) -> None:
        self._engine = ocrmac
        logger.info("Using ocrmac (Apple Vision ANE)")

    def _vision_box_to_bbox(
        self,
        vision_bbox: tuple[float, float, float, float],
        metadata: VideoMetadata,
    ) -> BoundingBox:
        x_norm, y_norm, w_norm, h_norm = vision_bbox
        return BoundingBox(
            x1=int(x_norm * metadata.width),
            y1=int((1 - y_norm - h_norm) * metadata.height),
            x2=int((x_norm + w_norm) * metadata.width),
            y2=int((1 - y_norm) * metadata.height),
        )

    def detect(self, frame: np.ndarray, metadata: VideoMetadata) -> list[Subtitle]:
        from PIL import Image

        H = metadata.height
        Y = frame[:H, :]
        pil_img = Image.fromarray(Y, mode="L").convert("RGB")

        results = OCR(
            pil_img,
            language_preference=["zh-Hans"],
            framework="vision",
            recognition_level="accurate",
        ).recognize()

        return [
            Subtitle(
                text=text,
                conf=conf,
                bbox=self._vision_box_to_bbox(vision_bbox=bbox, metadata=metadata),
                start=0.0,
                end=0.0,
            )
            for (text, conf, bbox) in results
        ]


def _is_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def ocr(
    input_path: str | Path,
    metadata: VideoMetadata,
    ocr_engine: OcrEngine,
    config: VideoConfig,
) -> list[Subtitle]:
    input_path = Path(input_path)

    subtitles: list[Subtitle] = []
    for frame_idx, frame_ts, frame_yuv in iter_frames(input_path, metadata):
        if frame_idx % 10 == 0:
            print(f"OCRed {frame_idx}/{metadata.total_frames}", end="\r")

        if frame_idx % config.ocr_sample_interval != 0:
            continue

        time.sleep(config.ocr_delay)

        ocr_result = ocr_engine.detect(frame=frame_yuv, metadata=metadata)

        for s in ocr_result:
            # skip low confidence text
            if s.conf < config.ocr_conf_threshold:
                continue

            # skip empty text
            text = s.text.strip()
            if not text:
                continue

            if config.ocr_chinese_only and not _is_chinese(text):
                continue

            subtitles.append(
                Subtitle(
                    text=text,
                    conf=s.conf,
                    bbox=replace(s.bbox),
                    start=frame_ts,
                    end=frame_ts,
                )
            )

    return subtitles
