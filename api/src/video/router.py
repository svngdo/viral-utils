import logging
import shutil
from collections.abc import Generator

from fastapi import APIRouter
from fastapi.sse import EventSourceResponse

from src.config import settings
from src.job.schemas import SSEEvent
from src.video import service
from src.video.dependencies import (
    InpaintConfigDep,
    InpaintEngineDep,
    OcrConfigDep,
    OcrEngineDep,
    SubtitleConfigDep,
    VideoEngineDep,
)
from src.video.schemas import ProcessVideosRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/video")


@router.post(path="/process/stream", response_class=EventSourceResponse)
def process_stream(
    request: ProcessVideosRequest,
    video_engine: VideoEngineDep,
    ocr_engine: OcrEngineDep,
    ocr_config: OcrConfigDep,
    subtitle_config: SubtitleConfigDep,
    inpaint_engine: InpaintEngineDep,
    inpaint_config: InpaintConfigDep,
) -> Generator[SSEEvent]:
    yield from service.process(
        request=request,
        video_engine=video_engine,
        ocr_engine=ocr_engine,
        ocr_config=ocr_config,
        subtitle_config=subtitle_config,
        inpaint_engine=inpaint_engine,
        inpaint_config=inpaint_config,
        llm_models=request.llm_models,
    )


@router.delete("/cache/clear")
def clear_cache_route():
    for p in settings.cache_dir.glob("*"):
        try:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.is_file():
                p.unlink()
        except Exception as e:
            logger.error(e)

    return {"status": "done"}
