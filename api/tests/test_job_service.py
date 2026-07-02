import asyncio
import threading

import pytest

from src.job import service
from src.job.exceptions import JobCancelled
from src.job.schemas import JobStatus, LogEvent, StatusEvent
from src.job.state import jobs


def sync_events():
    yield StatusEvent(status=JobStatus.COMPLETED)


def test_request_cancel_sets_event_and_cancelling_status():
    job = service.create_job()

    try:
        cancelled = service.request_cancel(job.id)

        assert cancelled is job
        assert job.cancel.is_set()
        assert job.status == JobStatus.CANCELLING
    finally:
        jobs.pop(job.id, None)


def test_request_cancel_keeps_terminal_status():
    job = service.create_job()
    job.status = JobStatus.COMPLETED

    try:
        cancelled = service.request_cancel(job.id)

        assert cancelled is job
        assert not job.cancel.is_set()
        assert job.status == JobStatus.COMPLETED
    finally:
        jobs.pop(job.id, None)


@pytest.mark.anyio
async def test_run_sync_job_accepts_sync_event_generator():
    job = service.create_job()

    try:
        await service._run_sync_job(job.id, sync_events())

        event = await jobs[job.id].queue.get()
        assert event.status == JobStatus.COMPLETED
        assert jobs[job.id].done is True
    finally:
        jobs.pop(job.id, None)


@pytest.mark.anyio
async def test_run_sync_job_tracks_status_events():
    job = service.create_job()

    try:
        await service._run_sync_job(job.id, sync_events())

        assert jobs[job.id].status == JobStatus.COMPLETED
    finally:
        jobs.pop(job.id, None)


@pytest.mark.anyio
async def test_run_sync_job_emits_cancelled_when_cancelled_generator_exits_quietly():
    job = service.create_job()
    job.cancel.set()

    def cancelled_events():
        yield LogEvent(message="stopping")

    try:
        await service._run_sync_job(job.id, cancelled_events())

        events = [await jobs[job.id].queue.get(), await jobs[job.id].queue.get()]
        assert events[0].message == "stopping"
        assert events[1].status == JobStatus.CANCELLED
        assert jobs[job.id].status == JobStatus.CANCELLED
        assert jobs[job.id].done is True
    finally:
        jobs.pop(job.id, None)


@pytest.mark.anyio
async def test_run_sync_job_emits_cancelled_when_generator_raises_job_cancelled():
    job = service.create_job()

    def cancelled_events():
        yield LogEvent(message="stopping")
        raise JobCancelled()

    try:
        await service._run_sync_job(job.id, cancelled_events())

        events = [await jobs[job.id].queue.get(), await jobs[job.id].queue.get()]
        assert events[0].message == "stopping"
        assert events[1].status == JobStatus.CANCELLED
        assert jobs[job.id].status == JobStatus.CANCELLED
        assert jobs[job.id].done is True
    finally:
        jobs.pop(job.id, None)


@pytest.mark.anyio
async def test_run_async_job_cancel_suppresses_late_completion_status():
    job = service.create_job()
    service.request_cancel(job.id)

    async def cancelled_events():
        yield StatusEvent(status=JobStatus.RUNNING)
        yield StatusEvent(status=JobStatus.COMPLETED)

    try:
        await service._run_async_job(job.id, cancelled_events())

        event = await jobs[job.id].queue.get()
        assert event.status == JobStatus.CANCELLED
        assert jobs[job.id].status == JobStatus.CANCELLED
        assert jobs[job.id].done is True
        assert jobs[job.id].queue.empty()
    finally:
        jobs.pop(job.id, None)


@pytest.mark.anyio
async def test_run_sync_job_cancel_suppresses_late_completion_status():
    job = service.create_job()
    service.request_cancel(job.id)

    def cancelled_events():
        yield StatusEvent(status=JobStatus.RUNNING)
        yield LogEvent(message="stopping")
        yield StatusEvent(status=JobStatus.COMPLETED)

    try:
        await service._run_sync_job(job.id, cancelled_events())

        events = [await jobs[job.id].queue.get(), await jobs[job.id].queue.get()]
        assert events[0].message == "stopping"
        assert events[1].status == JobStatus.CANCELLED
        assert jobs[job.id].status == JobStatus.CANCELLED
        assert jobs[job.id].done is True
        assert jobs[job.id].queue.empty()
    finally:
        jobs.pop(job.id, None)


@pytest.mark.anyio
async def test_run_sync_job_does_not_block_event_loop():
    job = service.create_job()
    release_generator = threading.Event()

    def blocking_events():
        yield LogEvent(message="started")
        release_generator.wait(timeout=5)
        yield StatusEvent(status=JobStatus.COMPLETED)

    task = asyncio.create_task(service._run_sync_job(job.id, blocking_events()))

    try:
        event = await asyncio.wait_for(jobs[job.id].queue.get(), timeout=1)
        assert event.type == "log"
        assert event.message == "started"

        release_generator.set()
        await asyncio.wait_for(task, timeout=1)
        assert jobs[job.id].done is True
    finally:
        release_generator.set()
        if not task.done():
            task.cancel()
        jobs.pop(job.id, None)
