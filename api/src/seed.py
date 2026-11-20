import asyncio

from sqlalchemy import text

from src.database import engine
from src.systems.constants import SYSTEMS
from src.systems.repository import create_system
from src.systems.schemas import SystemCreate


async def seed_db():
    try:
        async with engine.begin() as db:
            for item in SYSTEMS:
                existing = await db.scalar(
                    text("SELECT id FROM systems WHERE name = :name"),
                    {"name": item["name"]},
                )
                if existing is None:
                    await create_system(db, SystemCreate(**item))
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(seed_db())
