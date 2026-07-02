import asyncio
import logging
from asyncio import AbstractEventLoop
from collections.abc import AsyncIterable, Iterable
from uuid import uuid4

from aiosqlite import Connection
from starlette.concurrency import run_in_threadpool

from src.douyin import service as douyin_service
from src.inpainting.engine import InpaintEngineProtocol
from src.inpainting.schemas import InpaintConfig
from src.job.exceptions import JobCancelled
from src.job.schemas import (
    JobStatus,
    LogEvent,
    SSEEvent,
    StatusEvent,
)
from src.job.state import JobState, jobs
from src.llm.schemas import LLMModel
from src.ocr.engine import OcrEngine
from src.ocr.schemas import OcrConfig
from src.subtitle.schemas import SubtitleConfig
from src.tikhub.client import TikHubClient
from src.video import service as video_service
from src.video.engine import VideoEngineProtocol
from src.video.schemas import ProcessVideosRequest

logger = logging.getLogger(__name__)


def is_terminal_status(status: JobStatus) -> bool:
    """Return True when a job status is final and should not move again."""
    return status in {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }


def is_terminal_event(event: SSEEvent) -> bool:
    """Return True when an SSE event is a final job status update."""
    return isinstance(event, StatusEvent) and is_terminal_status(event.status)


def raise_if_cancelled(cancel) -> None:
    """Raise JobCancelled when a worker sees that cancellation was requested."""
    if cancel and cancel.is_set():
        raise JobCancelled()


def create_job():
    """Create a new in-memory job record and return its mutable state."""
    job_id = str(uuid4())
    state = JobState(id=job_id)
    jobs[job_id] = state
    return state


def get_job(job_id: str) -> JobState | None:
    """Look up the in-memory state for a job id."""
    return jobs.get(job_id)


def request_cancel(job_id: str) -> JobState | None:
    """Mark a running job as cancelling and signal its worker to stop."""
    state = get_job(job_id)
    if not state:
        return None

    if is_terminal_status(state.status):
        return state

    state.cancel.set()
    state.status = JobStatus.CANCELLING
    return state


def _should_publish_status(state: JobState, status: JobStatus) -> bool:
    """Suppress normal status updates after cancel; allow only final outcomes."""
    if not state.cancel.is_set():
        return True

    return status in {JobStatus.CANCELLED, JobStatus.FAILED}


def _record_event_status(state: JobState, event: SSEEvent) -> bool:
    """Apply status events to job state and say whether the event should emit."""
    if isinstance(event, StatusEvent):
        if not _should_publish_status(state, event.status):
            return False
        state.status = event.status
    return True


def _publish_sync_events(
    loop: AbstractEventLoop,
    state: JobState,
    events: Iterable[SSEEvent],
) -> None:
    """Publish events produced by sync code into the async SSE queue safely."""
    for event in events:
        if _record_event_status(state, event):
            loop.call_soon_threadsafe(state.queue.put_nowait, event)


async def _publish_async_event(state: JobState, event: SSEEvent) -> None:
    """Record an async event and enqueue it for SSE clients when allowed."""
    if _record_event_status(state, event):
        await state.queue.put(event)


async def _emit_failed(state: JobState, error: Exception) -> None:
    """Publish a failure log and final FAILED status for a crashed job."""
    state.error = str(error)
    await state.queue.put(LogEvent(message=str(error)))
    failed = StatusEvent(status=JobStatus.FAILED)
    state.status = failed.status
    await state.queue.put(failed)


async def _emit_cancelled(state: JobState) -> None:
    """Publish final CANCELLED status unless the job already failed."""
    if state.status == JobStatus.FAILED:
        return

    cancelled = StatusEvent(status=JobStatus.CANCELLED)
    state.status = cancelled.status
    await state.queue.put(cancelled)


async def _complete_job(state: JobState) -> None:
    """Finalize bookkeeping after a worker exits."""
    if state.cancel.is_set() and not is_terminal_status(state.status):
        await _emit_cancelled(state)
    state.done = True


async def _run_async_job(job_id: str, events: AsyncIterable[SSEEvent]) -> None:
    """Run an async event producer and translate its output into job state."""
    state = jobs[job_id]
    try:
        async for event in events:
            await _publish_async_event(state, event)
    except JobCancelled:
        await _emit_cancelled(state)
    except Exception as e:
        logger.exception("job failed - job_id=%s", job_id)
        await _emit_failed(state, e)
    finally:
        await _complete_job(state)


async def _run_sync_job(job_id: str, events: Iterable[SSEEvent]) -> None:
    """Run a sync event producer in a thread and publish events back safely."""
    state = jobs[job_id]
    loop = asyncio.get_running_loop()

    try:
        await run_in_threadpool(
            _publish_sync_events,
            loop,
            state,
            events,
        )
    except JobCancelled:
        await _emit_cancelled(state)
    except Exception as e:
        logger.exception("job failed - job_id=%s", job_id)
        await _emit_failed(state, e)
    finally:
        await _complete_job(state)


async def run_fetch_latest_videos_job(
    job_id: str,
    db: Connection,
    tikhub: TikHubClient,
) -> None:
    """Background job entrypoint for fetching latest Douyin videos."""

    await _run_async_job(
        job_id=job_id,
        events=douyin_service.fetch_latest_videos(
            db=db,
            tikhub=tikhub,
            cancel=jobs[job_id].cancel,
        ),
    )


async def run_fetch_user_videos_job(
    job_id: str,
    user_id: int,
    db: Connection,
    tikhub: TikHubClient,
) -> None:
    """Background job entrypoint for fetching videos from one Douyin user."""
    await _run_async_job(
        job_id=job_id,
        events=douyin_service.fetch_user_videos(
            user_id=user_id,
            db=db,
            tikhub=tikhub,
            cancel=jobs[job_id].cancel,
        ),
    )


async def run_process_videos_job(
    job_id: str,
    request: ProcessVideosRequest,
    video_engine: VideoEngineProtocol,
    ocr_engine: OcrEngine,
    ocr_config: OcrConfig,
    subtitle_config: SubtitleConfig,
    inpaint_engine: InpaintEngineProtocol,
    inpaint_config: InpaintConfig,
    llm_models: list[LLMModel],
) -> None:
    """Background job entrypoint for the sync video processing pipeline."""
    state = jobs[job_id]
    await _run_sync_job(
        job_id=job_id,
        events=video_service.process(
            request=request,
            video_engine=video_engine,
            ocr_engine=ocr_engine,
            ocr_config=ocr_config,
            subtitle_config=subtitle_config,
            inpaint_engine=inpaint_engine,
            inpaint_config=inpaint_config,
            llm_models=llm_models,
            cancel=state.cancel,
        ),
    )
