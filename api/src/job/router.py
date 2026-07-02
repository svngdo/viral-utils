import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.sse import EventSourceResponse

from src.database import DbConnection
from src.job import service as job_service
from src.job.schemas import (
    JobCancelResponse,
    JobCreateResponse,
    JobResponse,
    SSEEvent,
)
from src.tikhub.dependencies import TikHubClientDep
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

router = APIRouter(prefix="/jobs")


@router.get("/{job_id}")
async def get_job(job_id: str) -> JobResponse:
    """Return the current status for a single background job"""
    state = job_service.get_job(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(id=job_id, status=state.status, error=state.error)


@router.get("/{job_id}/events", response_class=EventSourceResponse)
async def get_job_events(job_id: str) -> AsyncGenerator[SSEEvent]:
    """Stream server-sent events for a running or recently completed job.

    Each job owns an async queue populated by the worker service. The response
    waits for events, yields them to the SSE client as they arrive, and closes
    the stream after the worker emits a terminal event such as completion,
    failure, or cancellation.
    """
    state = job_service.get_job(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")

    while True:
        event = await state.queue.get()
        yield event

        if job_service.is_terminal_event(event):
            break


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str) -> JobCancelResponse:
    """Request cancellation for a background job.

    Cancellation is cooperative: this marks the job state so the running worker
    can observe the request at its next cancellation checkpoint.
    """
    state = job_service.request_cancel(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobCancelResponse(id=job_id, status=state.status)


@router.post("/fetch-latest-videos")
async def create_fetch_latest_videos_job(
    background_tasks: BackgroundTasks,
    db: DbConnection,
    tikhub: TikHubClientDep,
):
    """Create a job that fetches the latest videos for tracked accounts.

    The endpoint allocates a job id synchronously, schedules the TikHub fetch
    worker on FastAPI's background task runner, and returns the URL clients can
    use to subscribe to progress events.
    """
    job = job_service.create_job()
    background_tasks.add_task(
        job_service.run_fetch_latest_videos_job,
        job.id,
        db,
        tikhub,
    )
    return JobCreateResponse(
        id=job.id,
        events_url=f"/jobs/{job.id}/events",
    )


@router.post("/users/{user_id}/fetch-videos")
async def create_fetch_user_videos_job(
    user_id: int,
    background_tasks: BackgroundTasks,
    db: DbConnection,
    tikhub: TikHubClientDep,
):
    """Create a job that fetches videos for one TikHub user."""
    job = job_service.create_job()
    background_tasks.add_task(
        job_service.run_fetch_user_videos_job,
        job.id,
        user_id,
        db,
        tikhub,
    )
    return JobCreateResponse(
        id=job.id,
        events_url=f"/jobs/{job.id}/events",
    )


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
    """Create a job that processes videos with OCR, subtitles, and inpainting.

    The request body selects the videos and processing options, while dependency
    injection supplies the concrete processing engines and configuration. The
    worker receives those objects plus any requested LLM model selections, then
    publishes progress and terminal events under the returned job id.
    """
    job = job_service.create_job()
    background_tasks.add_task(
        job_service.run_process_videos_job,
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
