from typing import Annotated

from fastapi import Depends, HTTPException

from src.inpainting.engine import InpaintEngineProtocol
from src.inpainting.engines.opencv import OpenCV
from src.inpainting.schemas import InpaintConfig, InpaintEngine
from src.ocr.engine import OcrEngine
from src.ocr.engines.ocrmac import Ocrmac
from src.ocr.engines.paddleocr import PaddleOcr
from src.ocr.schemas import OcrConfig
from src.subtitle.schemas import SubtitleConfig
from src.video.engine import VideoEngineProtocol
from src.video.engines.ffmpeg import FFmpeg
from src.video.schemas import ProcessVideosRequest, VideoConfig, VideoEngine


def get_video_engine(payload: ProcessVideosRequest) -> VideoEngineProtocol:
    config = VideoConfig(
        codec=payload.video_codec,
        quality=payload.video_quality,
    )
    match payload.video_engine:
        case VideoEngine.FFMPEG:
            return FFmpeg(config)
        case _:
            raise HTTPException(
                status_code=400, detail=f"Unknown video engine: {payload.video_engine}"
            )


def get_ocr_engine(payload: ProcessVideosRequest) -> OcrEngine:
    match payload.ocr_engine:
        case "ocrmac":
            return Ocrmac()
        case "paddleocr":
            return PaddleOcr()
        case _:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown OCR engine: {payload.ocr_engine}",
            )


def get_ocr_config(payload: ProcessVideosRequest) -> OcrConfig:
    return OcrConfig(
        sample_interval=payload.ocr_sample_interval,
        delay=payload.ocr_delay,
        chinese_only=payload.ocr_chinese_only,
    )


def get_subtitle_config(payload: ProcessVideosRequest) -> SubtitleConfig:
    return SubtitleConfig(
        time_gap_tolerance=payload.sub_time_gap_tolerance,
        text_similarity_threshold=payload.sub_text_similarity_threshold,
        box_iou_threshold=payload.sub_box_iou_threshold,
        frame_padding=payload.sub_frame_padding,
    )


def get_inpaint_engine(payload: ProcessVideosRequest) -> InpaintEngineProtocol:
    config = InpaintConfig(
        conf_threshold=payload.inpaint_conf_threshold,
        scale=payload.inpaint_scale,
        expand=payload.inpaint_expand,
        radius=payload.inpaint_radius,
        delay=payload.inpaint_delay,
    )
    match payload.inpaint_engine:
        case InpaintEngine.OPENCV:
            return OpenCV(config)
        # TODO: add lama
        case _:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown inpaint engine: {payload.inpaint_engine}",
            )


def get_inpaint_config(payload: ProcessVideosRequest) -> InpaintConfig:
    return InpaintConfig(
        conf_threshold=payload.inpaint_conf_threshold,
        scale=payload.inpaint_scale,
        expand=payload.inpaint_expand,
        radius=payload.inpaint_radius,
        delay=payload.inpaint_delay,
    )


VideoEngineDep = Annotated[VideoEngineProtocol, Depends(get_video_engine)]
OcrEngineDep = Annotated[OcrEngine, Depends(get_ocr_engine)]
OcrConfigDep = Annotated[OcrConfig, Depends(get_ocr_config)]
SubtitleConfigDep = Annotated[SubtitleConfig, Depends(get_subtitle_config)]
InpaintEngineDep = Annotated[InpaintEngineProtocol, Depends(get_inpaint_engine)]
InpaintConfigDep = Annotated[InpaintConfig, Depends(get_inpaint_config)]
