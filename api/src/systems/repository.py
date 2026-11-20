from uuid import UUID, uuid4

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncConnection

from src.systems.schemas import (
    System,
    SystemOrderBy,
    SystemRequest,
)
from src.utils.text import build_search_text


async def create_system(
    db: AsyncConnection,
    data: SystemRequest,
) -> System:
    system_id = uuid4()
    search_text = build_search_text(
        " ".join(filter(None, [data.name, data.description]))
    )

    result = await db.execute(
        text("""
            INSERT INTO systems (
                id,
                name,
                description,
                search_text
            )
            VALUES (
                :id,
                :name,
                :description,
                :search_text
            )
            RETURNING
                id,
                name,
                description,
                search_text,
                created_at,
                updated_at
         """),
        {
            "id": str(system_id),
            "name": data.name,
            "description": data.description,
            "search_text": search_text,
        },
    )

    return System.model_validate(result.mappings().one())


async def get_system_by_id(
    db: AsyncConnection,
    system_id: UUID,
) -> System | None:
    result = await db.execute(
        text("""
            SELECT
                id,
                name,
                description,
                search_text,
                created_at,
                updated_at
            FROM systems
            WHERE id = :id
        """),
        {
            "id": str(system_id),
        },
    )

    row = result.mappings().one_or_none()

    return System.model_validate(row) if row else None


async def get_system_by_name(
    db: AsyncConnection,
    name: str,
) -> System | None:
    result = await db.execute(
        text("""
            SELECT
                id,
                name,
                description,
                search_text,
                created_at,
                updated_at
            FROM systems
            WHERE name = :name COLLATE NOCASE
        """),
        {
            "name": str(name),
        },
    )

    row = result.mappings().one_or_none()

    return System.model_validate(row) if row else None


async def count_systems(db: AsyncConnection, search_query: str) -> int:
    result = await db.execute(
        text("""
            SELECT COUNT(*)
            FROM systems
            WHERE search_text LIKE :search_query
        """),
        {"search_query": f"%{build_search_text(search_query)}%"},
    )

    return result.scalar_one()


async def list_systems(
    db: AsyncConnection,
    search_query: str,
    order_by: list[SystemOrderBy],
    limit: int,
    offset: int,
) -> list[System]:
    order_by_str = (", ").join(
        [f"{order.id} {'DESC' if order.desc else 'ASC'}" for order in order_by]
    )

    result = await db.execute(
        text(f"""
            SELECT
                id,
                name,
                description,
                search_text,
                created_at,
                updated_at
            FROM systems
            WHERE search_text LIKE :search_query
            ORDER BY {order_by_str}, id ASC
            LIMIT :limit
            OFFSET :offset
         """),
        {
            "search_query": f"%{build_search_text(search_query)}%",
            "limit": limit,
            "offset": offset,
        },
    )

    rows = result.mappings().all()

    return [System.model_validate(row) for row in rows]


async def update_system(
    db: AsyncConnection,
    system_id: UUID,
    data: SystemRequest,
) -> System | None:
    search_text = build_search_text(
        " ".join(filter(None, [data.name, data.description]))
    )

    result = await db.execute(
        text("""
            UPDATE systems
            SET
                name = :name,
                description = :description,
                search_text = :search_text,
                updated_at = CURRENT_TIMESTAMP
            WHERE
                id = :id
            RETURNING
                id,
                name,
                description,
                search_text,
                created_at,
                updated_at
        """),
        {
            "id": str(system_id),
            "name": data.name,
            "description": data.description,
            "search_text": search_text,
        },
    )

    row = result.mappings().one_or_none()

    return System.model_validate(row) if row else None


async def delete_systems(
    db: AsyncConnection,
    system_ids: list[UUID],
) -> int:
    if not system_ids:
        return 0

    stmt = text("""
        DELETE FROM systems
        WHERE id IN :ids
    """).bindparams(bindparam("ids", expanding=True))

    result = await db.execute(stmt, {"ids": [str(i) for i in system_ids]})

    return result.rowcount
