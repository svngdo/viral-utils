import logging

import numpy as np
from ocrmac import ocrmac
from PIL import Image

from src.ocr.engine import OcrEngine
from src.subtitle.schemas import BoundingBox, Subtitle
from src.video.engine import VideoMetadata

logger = logging.getLogger(__name__)


class Ocrmac(OcrEngine):
    def __init__(self) -> None:
        self._engine = ocrmac

    def _vision_box_to_bbox(
        self,
        vision_bbox: tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
    ) -> BoundingBox:
        x_norm, y_norm, w_norm, h_norm = vision_bbox
        return BoundingBox(
            x1=int(x_norm * frame_width),
            y1=int((1 - y_norm - h_norm) * frame_height),
            x2=int((x_norm + w_norm) * frame_width),
            y2=int((1 - y_norm) * frame_height),
        )

    def extract(self, frame: np.ndarray, meta: VideoMetadata) -> list[Subtitle]:
        Y = frame[: meta.height]
        pil_img = Image.fromarray(Y, mode="L").convert("RGB")

        results = self._engine.OCR(
            pil_img,
            language_preference=["zh-Hans", "en-US"],
            framework="vision",
            recognition_level="accurate",
        ).recognize()

        return [
            Subtitle(
                text=text,
                conf=conf,
                bbox=self._vision_box_to_bbox(
                    vision_bbox=bbox,
                    frame_width=meta.width,
                    frame_height=meta.height,
                ),
                start=0.0,
                end=0.0,
            )
            for (text, conf, bbox) in results
        ]
