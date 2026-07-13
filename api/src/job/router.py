import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.sse import EventSourceResponse

from src.job import service as job_service
from src.job.douyin.router import router as douyin_job_router
from src.job.schemas import (
    JobCancelResponse,
    JobResponse,
    SSEEvent,
)
from src.job.video.router import router as video_job_router

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


router.include_router(douyin_job_router)
router.include_router(video_job_router)
