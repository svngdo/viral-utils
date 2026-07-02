import asyncio
import threading
from dataclasses import dataclass, field

from src.job.schemas import JobStatus, SSEEvent


@dataclass
class JobState:
    id: str
    queue: asyncio.Queue[SSEEvent] = field(default_factory=asyncio.Queue)
    cancel: threading.Event = field(default_factory=threading.Event)
    status: JobStatus = JobStatus.QUEUED
    done: bool = False
    error: str | None = None


jobs: dict[str, JobState] = {}
