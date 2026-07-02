from enum import StrEnum
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EventBase(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))


class StatusEvent(EventBase):
    type: Literal["status"] = "status"
    status: JobStatus


class ProgressEvent(EventBase):
    type: Literal["progress"] = "progress"
    done: int
    total: int


class LogEvent(EventBase):
    type: Literal["log"] = "log"
    message: str


type SSEEvent = Annotated[
    StatusEvent | ProgressEvent | LogEvent,
    Field(discriminator="type"),
]


class JobCreateResponse(BaseModel):
    id: str
    events_url: str


class JobCancelResponse(BaseModel):
    id: str
    status: JobStatus


class JobResponse(BaseModel):
    id: str
    status: JobStatus
    error: str | None = None
