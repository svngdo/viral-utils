from typing import Annotated, cast

from fastapi import HTTPException, Query

from src.systems.schemas import VALID_ORDER_COLUMNS, OrderByColumns, SystemOrderBy


def parse_order_by(order_by: Annotated[str, Query()]) -> list[SystemOrderBy]:
    result = []
    for item in order_by.split(","):
        parts = item.strip().split()

        if not parts or len(parts) > 2:
            raise ValueError(f"Invalid sorting expression: {item}")

        column = parts[0]
        direction = parts[1].lower() if len(parts) == 2 else "asc"

        if column not in VALID_ORDER_COLUMNS:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid sorting column: {column}",
            )

        if direction not in ("asc", "desc"):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid sorting direction: {direction}",
            )

        result.append(
            SystemOrderBy(
                id=cast(OrderByColumns, column),
                desc=direction == "desc",
            )
        )

    return result
