import asyncio
from pathlib import Path

import aiosqlite

from src.douyin.repository import (
    count_videos,
    delete_user_by_id,
    delete_video_by_id,
    insert_user,
    insert_video,
    select_user_by_id,
    select_users_to_fetch,
    select_video_by_id,
    select_videos_by_user_id,
    select_videos_page,
    select_videos_to_download,
    select_videos_to_download_by_user_ids,
    update_video_by_id,
    upsert_video,
)
from src.douyin.schemas import UserCreate, VideoCreate, VideoUpdate

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "migrations" / "schemas"


async def _connect_db(tmp_path):
    db_path = tmp_path / "test.db"
    conn = await aiosqlite.connect(db_path)
    await conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = aiosqlite.Row
    for schema in sorted(SCHEMA_DIR.glob("*.sql")):
        await conn.executescript(schema.read_text())
    await conn.commit()
    return conn


def test_select_users_to_fetch_filters_by_status_and_last_fetched(tmp_path):
    async def run():
        db = await _connect_db(tmp_path)
        try:
            await db.executemany(
                """
                INSERT INTO users (sec_uid, status, last_fetched)
                VALUES (:sec_uid, :status, :last_fetched)
                """,
                [
                    {
                        "sec_uid": "active-never",
                        "status": "active",
                        "last_fetched": None,
                    },
                    {"sec_uid": "active-old", "status": "active", "last_fetched": 1},
                    {
                        "sec_uid": "active-current",
                        "status": "active",
                        "last_fetched": 4102444800,
                    },
                    {
                        "sec_uid": "testing-never",
                        "status": "testing",
                        "last_fetched": None,
                    },
                    {"sec_uid": "testing-old", "status": "testing", "last_fetched": 1},
                    {"sec_uid": "pending", "status": "pending", "last_fetched": None},
                    {"sec_uid": "dropped", "status": "dropped", "last_fetched": None},
                ],
            )
            await db.commit()

            users = await select_users_to_fetch(db)

            assert {user.sec_uid for user in users} == {
                "active-never",
                "active-old",
                "testing-never",
            }
        finally:
            await db.close()

    asyncio.run(run())


def test_insert_user_persists_system_id(tmp_path):
    async def run():
        db = await _connect_db(tmp_path)
        try:
            await db.execute(
                "INSERT INTO systems (name, description) VALUES (:name, :description)",
                {"name": "System A", "description": None},
            )
            await db.commit()
            system_id = (await (await db.execute("SELECT id FROM systems")).fetchone())[
                "id"
            ]

            user = await insert_user(
                UserCreate(
                    sec_uid="user-with-system",
                    status="active",
                    system_id=system_id,
                ),
                db,
            )

            assert user is not None
            assert user.system_id == system_id
        finally:
            await db.close()

    asyncio.run(run())


def test_video_update_persists_metadata_and_download_selection(tmp_path):
    async def run():
        db = await _connect_db(tmp_path)
        try:
            await db.execute(
                "INSERT INTO users (sec_uid, status) VALUES (:sec_uid, :status)",
                {"sec_uid": "user-1", "status": "active"},
            )
            await db.commit()
            user_id = (await (await db.execute("SELECT id FROM users")).fetchone())[
                "id"
            ]

            video = await insert_video(
                VideoCreate(
                    aweme_id="video-1",
                    title="old title",
                    translated_title="old translated",
                    create_time=1,
                    digg_count=1,
                    duration=10,
                    urls='["old"]',
                    is_downloaded=False,
                    user_id=user_id,
                ),
                db,
            )
            assert video is not None

            updated = await update_video_by_id(
                video.id,
                VideoUpdate(
                    title="new title",
                    translated_title="new translated",
                    digg_count=99,
                    duration=20,
                    urls='["new"]',
                    is_downloaded=False,
                ),
                db,
            )

            assert updated is not None
            assert updated.title == "new title"
            assert updated.translated_title == "new translated"
            assert updated.digg_count == 99
            assert updated.duration == 20
            assert updated.urls == '["new"]'

            videos = await select_videos_to_download(db)
            assert [item.aweme_id for item in videos] == ["video-1"]
        finally:
            await db.close()

    asyncio.run(run())


def test_upsert_video_preserves_existing_download_state(tmp_path):
    async def run():
        db = await _connect_db(tmp_path)
        try:
            user = await insert_user(UserCreate(sec_uid="user-1", status="active"), db)
            assert user is not None

            video = await insert_video(
                VideoCreate(
                    aweme_id="video-1",
                    title="old title",
                    translated_title="old translated",
                    create_time=1,
                    digg_count=1,
                    duration=10,
                    urls='["old"]',
                    is_downloaded=True,
                    user_id=user.id,
                ),
                db,
            )
            assert video is not None

            upserted = await upsert_video(
                VideoCreate(
                    aweme_id="video-1",
                    title="new title",
                    translated_title="new translated",
                    create_time=2,
                    digg_count=99,
                    duration=20,
                    urls='["new"]',
                    is_downloaded=False,
                    user_id=user.id,
                ),
                db,
            )

            assert upserted is not None
            assert upserted.title == "new title"
            assert upserted.translated_title == "new translated"
            assert upserted.create_time == 2
            assert upserted.digg_count == 99
            assert upserted.duration == 20
            assert upserted.urls == '["new"]'
            assert upserted.is_downloaded is True
        finally:
            await db.close()

    asyncio.run(run())


def test_select_videos_page_orders_by_create_time_and_id(tmp_path):
    async def run():
        db = await _connect_db(tmp_path)
        try:
            user = await insert_user(UserCreate(sec_uid="user-1", status="active"), db)
            assert user is not None

            for aweme_id, create_time in [
                ("old", 10),
                ("newer-a", 20),
                ("newer-b", 20),
                ("newest", 30),
            ]:
                video = await insert_video(
                    VideoCreate(
                        aweme_id=aweme_id,
                        title=None,
                        translated_title=None,
                        create_time=create_time,
                        digg_count=1,
                        duration=None,
                        urls=None,
                        is_downloaded=False,
                        user_id=user.id,
                    ),
                    db,
                )
                assert video is not None

            first_page = await select_videos_page(limit=2, offset=0, db=db)
            second_page = await select_videos_page(limit=2, offset=2, db=db)

            assert [video.aweme_id for video in first_page] == ["newest", "newer-b"]
            assert [video.aweme_id for video in second_page] == ["newer-a", "old"]
        finally:
            await db.close()

    asyncio.run(run())


def test_select_videos_to_download_by_user_ids_filters_selected_pending_videos(
    tmp_path,
):
    async def run():
        db = await _connect_db(tmp_path)
        try:
            selected_user = await insert_user(
                UserCreate(sec_uid="selected", status="dropped"),
                db,
            )
            other_user = await insert_user(
                UserCreate(sec_uid="other", status="active"),
                db,
            )
            assert selected_user is not None
            assert other_user is not None

            for aweme_id, user_id, is_downloaded in [
                ("selected-pending", selected_user.id, False),
                ("selected-downloaded", selected_user.id, True),
                ("other-pending", other_user.id, False),
            ]:
                video = await insert_video(
                    VideoCreate(
                        aweme_id=aweme_id,
                        title=None,
                        translated_title=None,
                        create_time=1,
                        digg_count=1,
                        duration=None,
                        urls=None,
                        is_downloaded=is_downloaded,
                        user_id=user_id,
                    ),
                    db,
                )
                assert video is not None

            videos = await select_videos_to_download_by_user_ids(
                [selected_user.id],
                db,
            )

            assert [video.aweme_id for video in videos] == ["selected-pending"]
        finally:
            await db.close()

    asyncio.run(run())


def test_count_videos_returns_total_video_count(tmp_path):
    async def run():
        db = await _connect_db(tmp_path)
        try:
            user = await insert_user(UserCreate(sec_uid="user-1", status="active"), db)
            assert user is not None

            for index in range(3):
                video = await insert_video(
                    VideoCreate(
                        aweme_id=f"video-{index}",
                        title=None,
                        translated_title=None,
                        create_time=index,
                        digg_count=1,
                        duration=None,
                        urls=None,
                        is_downloaded=False,
                        user_id=user.id,
                    ),
                    db,
                )
                assert video is not None

            assert await count_videos(db) == 3
        finally:
            await db.close()

    asyncio.run(run())


def test_delete_user_cascades_videos(tmp_path):
    async def run():
        db = await _connect_db(tmp_path)
        try:
            user = await insert_user(UserCreate(sec_uid="user-1", status="active"), db)
            assert user is not None

            video = await insert_video(
                VideoCreate(
                    aweme_id="video-1",
                    title="title",
                    translated_title=None,
                    create_time=1,
                    digg_count=1,
                    duration=None,
                    urls=None,
                    is_downloaded=False,
                    user_id=user.id,
                ),
                db,
            )
            assert video is not None

            deleted = await delete_user_by_id(user.id, db)

            assert deleted is True
            assert await select_user_by_id(user.id, db) is None
            assert await select_videos_by_user_id(user.id, db) == []
        finally:
            await db.close()

    asyncio.run(run())


def test_delete_video_removes_only_video(tmp_path):
    async def run():
        db = await _connect_db(tmp_path)
        try:
            user = await insert_user(UserCreate(sec_uid="user-1", status="active"), db)
            assert user is not None

            video = await insert_video(
                VideoCreate(
                    aweme_id="video-1",
                    title=None,
                    translated_title=None,
                    create_time=1,
                    digg_count=1,
                    duration=None,
                    urls=None,
                    is_downloaded=False,
                    user_id=user.id,
                ),
                db,
            )
            assert video is not None

            deleted = await delete_video_by_id(video.id, db)

            assert deleted is True
            assert await select_user_by_id(user.id, db) is not None
            assert await select_video_by_id(video.id, db) is None
        finally:
            await db.close()

    asyncio.run(run())
