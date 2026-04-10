import time
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import numpy as np

from src.logging import get_logger
from src.video.config import VideoConfig
from src.video.ffmpeg import iter_frames
from src.video.types import BoundingBox, Subtitle, VideoMetadata

logger = get_logger(__name__)


class OcrEngine(Protocol):
    def detect(
        self,
        frame: np.ndarray,
        metadata: VideoMetadata | None = None,
    ) -> list[Subtitle]:
        """Detect text in a single frame and return a list of Subtitle."""
        ...


class Ocrmac(OcrEngine):
    def __init__(self) -> None:
        from ocrmac import ocrmac

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
        from ocrmac.ocrmac import OCR
        from PIL import Image

        Y = frame[: metadata.height]
        pil_img = Image.fromarray(Y, mode="L").convert("RGB")

        results = OCR(
            pil_img,
            language_preference=["zh-Hans", "en-US"],
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


class PaddleOcr(OcrEngine):
    def __init__(self):
        from paddleocr import PaddleOCR

        self._engine = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,  # no detect page rotation
            use_doc_unwarping=False,  # no straighten curved pages
            use_textline_orientation=False,  # no detect text line angles
        )

    def detect(self, frame: np.ndarray, metadata: VideoMetadata) -> list[Subtitle]:
        import cv2

        gray = frame[: metadata.height]
        # gray = cv2.resize(gray, (gray.shape[1] * 3 // 4, gray.shape[0] * 3 // 4))
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        result = self._engine.predict(rgb)
        boxes = []

        if not result:
            return boxes

        for res in result:
            for text, confidence, (x1, y1, x2, y2) in zip(
                res["rec_texts"],
                res["rec_scores"],
                res["rec_boxes"],
            ):
                if not text.strip():
                    continue

                boxes.append(
                    Subtitle(
                        text=text,
                        bbox=BoundingBox(
                            x1=int(x1),
                            y1=int(y1),
                            x2=int(x2),
                            y2=int(y2),
                        ),
                        conf=float(confidence),
                        start=0.0,
                        end=0.0,
                    )
                )

        return boxes


def _is_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _get_ocr_engine(engine_name: str) -> OcrEngine:
    if engine_name not in {"ocrmac", "paddleocr"}:
        raise ValueError(f"Invalid ocr engine name: {engine_name}")

    match engine_name:
        case "ocrmac":
            return Ocrmac()
        case _:
            return PaddleOcr()


def ocr(
    input_path: str | Path,
    metadata: VideoMetadata,
    config: VideoConfig,
) -> list[Subtitle]:
    input_path = Path(input_path)
    ocr_engine = _get_ocr_engine(config.ocr_engine_name)
    subtitles: list[Subtitle] = []

    total_frames = int(metadata.total_frames / config.ocr_sample_interval)
    scanned_frames = 0

    for frame_idx, frame_ts, frame_yuv in iter_frames(input_path, metadata):
        if frame_idx % config.ocr_sample_interval != 0:
            continue

        time.sleep(config.ocr_delay)

        results = ocr_engine.detect(frame=frame_yuv, metadata=metadata)

        for r in results:
            if config.ocr_chinese_only and not _is_chinese(r.text):
                continue

            subtitles.append(
                Subtitle(
                    text=r.text,
                    conf=r.conf,
                    bbox=replace(r.bbox),
                    start=frame_ts,
                    end=frame_ts,
                )
            )

        scanned_frames += 1
        print(f"OCRed {scanned_frames}/{total_frames}", end="\r")

    logger.info(f"OCRed {scanned_frames} frames")
    return subtitles
