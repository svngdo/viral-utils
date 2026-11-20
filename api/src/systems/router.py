from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Path, Query, Response, status
from sqlalchemy.exc import IntegrityError

from src.database import DB
from src.systems import repository as systems_repo
from src.systems.dependencies import parse_order_by
from src.systems.exceptions import (
    NoSystemsFoundError,
    SystemNameConflictError,
    SystemNotFoundError,
)
from src.systems.schemas import (
    BulkDeleteRequest,
    System,
    SystemListResponse,
    SystemOrderBy,
    SystemRequest,
)

router = APIRouter(prefix="/systems")


@router.post("")
async def create_system(
    db: DB,
    *,
    request: SystemRequest,
) -> System:
    existing = await systems_repo.get_system_by_name(db, request.name)
    if existing:
        raise SystemNameConflictError()

    return await systems_repo.create_system(db, request)


@router.get("/{system_id}")
async def get_system(
    db: DB,
    *,
    system_id: Annotated[UUID, Path()],
) -> System:
    system = await systems_repo.get_system_by_id(db, system_id)
    if system is None:
        raise SystemNotFoundError()
    return system


@router.get("")
async def list_systems(
    db: DB,
    *,
    q: str = "",
    order_by: Annotated[list[SystemOrderBy], Depends(parse_order_by)],
    limit: Annotated[int, Query(ge=1, le=100)],
    offset: Annotated[int, Query(ge=0)],
) -> SystemListResponse:
    systems = await systems_repo.list_systems(
        db,
        search_query=q,
        order_by=order_by,
        limit=limit,
        offset=offset,
    )
    system_count = await systems_repo.count_systems(db, search_query=q)

    return SystemListResponse(items=systems, total=system_count)


@router.patch("/{system_id}")
async def update_system(
    db: DB,
    *,
    system_id: UUID,
    request: SystemRequest,
) -> System:
    try:
        updated = await systems_repo.update_system(db, system_id, request)
    except IntegrityError:
        raise SystemNameConflictError() from None

    if updated is None:
        raise SystemNotFoundError()

    return updated


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def delete_system(
    db: DB,
    *,
    body: BulkDeleteRequest,
) -> Response:
    deleted_count = await systems_repo.delete_systems(db, body.ids)

    if deleted_count == 0:
        raise NoSystemsFoundError()

    return Response(status_code=status.HTTP_204_NO_CONTENT)
