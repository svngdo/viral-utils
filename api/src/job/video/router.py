from fastapi import APIRouter, BackgroundTasks, Request

from src.job import service as job_service
from src.job.schemas import JobCreateResponse
from src.job.video import service as video_job_service
from src.video.dependencies import (
    InpaintConfigDep,
    InpaintEngineDep,
    OcrConfigDep,
    OcrEngineDep,
    SubtitleConfigDep,
    VideoEngineDep,
)
from src.video.schemas import ProcessVideosRequest

router = APIRouter(prefix="/video")


@router.post("/process-videos")
async def create_process_videos_job(
    request: Request,
    payload: ProcessVideosRequest,
    video_engine: VideoEngineDep,
    ocr_engine: OcrEngineDep,
    ocr_config: OcrConfigDep,
    subtitle_config: SubtitleConfigDep,
    inpaint_engine: InpaintEngineDep,
    inpaint_config: InpaintConfigDep,
    background_tasks: BackgroundTasks,
):
    """Create a job that processes videos with OCR, subtitles, and inpainting."""
    job = job_service.create_job()
    background_tasks.add_task(
        video_job_service.run_process_videos_job,
        job.id,
        payload,
        video_engine,
        ocr_engine,
        ocr_config,
        subtitle_config,
        inpaint_engine,
        inpaint_config,
        payload.llm_models,
    )
    return JobCreateResponse(
        id=job.id,
        events_url=str(request.url_for("get_job_events", job_id=job.id)),
    )
