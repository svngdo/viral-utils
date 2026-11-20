from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from pydantic import UUID4, BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from src.types import SystemName

VALID_ORDER_COLUMNS: set[str] = {
    "id",
    "name",
    "description",
    "created_at",
    "updated_at",
}

OrderByColumns = Literal[
    "id",
    "name",
    "description",
    "created_at",
    "updated_at",
]


class SystemOrderBy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: OrderByColumns
    desc: bool


SystemSortingList = TypeAdapter(list[SystemOrderBy])


class System(BaseModel):
    id: UUID4
    name: SystemName
    description: str | None
    search_text: str | None
    created_at: datetime
    updated_at: datetime

    @field_validator("created_at", "updated_at")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class SystemRequest(BaseModel):
    name: SystemName
    description: str | None


class SystemListResponse(BaseModel):
    items: list[System]
    total: int


class BulkDeleteRequest(BaseModel):
    ids: list[UUID] = Field(min_length=1, max_length=500)
