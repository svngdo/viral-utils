import logging

from aiosqlite import Connection

from src.douyin.queries import (
    COUNT_VIDEOS,
    DELETE_USER_BY_ID,
    DELETE_VIDEO_BY_ID,
    INSERT_USER,
    INSERT_VIDEO,
    SELECT_SYSTEM_NAME_BY_VIDEO_ID,
    SELECT_USER_BY_ID,
    SELECT_USER_BY_SEC_UID,
    SELECT_USERS,
    SELECT_USERS_TO_FETCH,
    SELECT_VIDEO_BY_AWEME_ID,
    SELECT_VIDEO_BY_ID,
    SELECT_VIDEOS,
    SELECT_VIDEOS_BY_USER_ID,
    SELECT_VIDEOS_PAGE,
    SELECT_VIDEOS_TO_DOWNLOAD,
    SELECT_VIDEOS_TO_DOWNLOAD_BY_USER_IDS,
    UPDATE_USER_BY_ID,
    UPDATE_VIDEO_BY_ID,
    UPSERT_USER,
    UPSERT_VIDEO,
)
from src.douyin.schemas import (
    User,
    UserCreate,
    UserUpdate,
    Video,
    VideoCreate,
    VideoUpdate,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# USER
# ==============================================================================


async def select_users(db: Connection) -> list[User]:
    cur = await db.execute(SELECT_USERS)
    rows = await cur.fetchall()
    return [User.model_validate(dict(row)) for row in rows]


async def select_users_to_fetch(db: Connection) -> list[User]:
    cur = await db.execute(SELECT_USERS_TO_FETCH)
    rows = await cur.fetchall()
    return [User.model_validate(dict(row)) for row in rows]


async def select_user_by_id(user_id: int, db: Connection) -> User | None:
    cur = await db.execute(SELECT_USER_BY_ID, {"id": user_id})
    row = await cur.fetchone()
    return User.model_validate(dict(row)) if row else None


async def select_user_by_sec_uid(sec_uid: str, db: Connection) -> User | None:
    cur = await db.execute(SELECT_USER_BY_SEC_UID, {"sec_uid": sec_uid})
    row = await cur.fetchone()
    return User.model_validate(dict(row)) if row else None


async def insert_user(user: UserCreate, db: Connection) -> User | None:
    await db.execute(INSERT_USER, user.model_dump())
    await db.commit()
    return await select_user_by_sec_uid(sec_uid=user.sec_uid, db=db)


async def update_user_by_id(
    user_id: int, user: UserUpdate, db: Connection
) -> User | None:
    existing = await select_user_by_id(user_id=user_id, db=db)
    if not existing:
        return None

    updated = existing.model_copy(update=user.model_dump(exclude_unset=True))
    await db.execute(UPDATE_USER_BY_ID, updated.model_dump())
    await db.commit()
    return updated


async def upsert_user(user: UserCreate, db: Connection) -> User | None:
    await db.execute(UPSERT_USER, user.model_dump())
    await db.commit()
    return await select_user_by_sec_uid(sec_uid=user.sec_uid, db=db)


async def delete_user_by_id(user_id: int, db: Connection) -> bool:
    existing = await select_user_by_id(user_id=user_id, db=db)
    if not existing:
        return False

    await db.execute(DELETE_USER_BY_ID, {"id": user_id})
    await db.commit()
    return True


# ==============================================================================
# VIDEO
# ==============================================================================


async def select_videos(db: Connection) -> list[Video]:
    cur = await db.execute(SELECT_VIDEOS)
    rows = await cur.fetchall()
    return [Video.model_validate(dict(row)) for row in rows]


async def count_videos(db: Connection) -> int:
    cur = await db.execute(COUNT_VIDEOS)
    row = await cur.fetchone()
    if row is None:
        raise RuntimeError("COUNT_VIDEOS returned no row")
    return int(row["total"])


async def select_videos_page(limit: int, offset: int, db: Connection) -> list[Video]:
    cur = await db.execute(SELECT_VIDEOS_PAGE, {"limit": limit, "offset": offset})
    rows = await cur.fetchall()
    return [Video.model_validate(dict(row)) for row in rows]


async def select_videos_to_download(db: Connection) -> list[Video]:
    cur = await db.execute(SELECT_VIDEOS_TO_DOWNLOAD)
    rows = await cur.fetchall()
    return [Video.model_validate(dict(row)) for row in rows]


async def select_videos_to_download_by_user_ids(
    user_ids: list[int], db: Connection
) -> list[Video]:
    if not user_ids:
        return []

    placeholders = ", ".join(f":user_id_{index}" for index in range(len(user_ids)))
    params = {f"user_id_{index}": user_id for index, user_id in enumerate(user_ids)}
    cur = await db.execute(
        SELECT_VIDEOS_TO_DOWNLOAD_BY_USER_IDS.format(placeholders=placeholders),
        params,
    )
    rows = await cur.fetchall()
    return [Video.model_validate(dict(row)) for row in rows]


async def select_video_by_id(video_id: int, db: Connection) -> Video | None:
    cur = await db.execute(SELECT_VIDEO_BY_ID, {"id": video_id})
    row = await cur.fetchone()
    return Video.model_validate(dict(row)) if row else None


async def select_video_by_aweme_id(aweme_id: str, db: Connection) -> Video | None:
    cur = await db.execute(SELECT_VIDEO_BY_AWEME_ID, {"aweme_id": aweme_id})
    row = await cur.fetchone()
    return Video.model_validate(dict(row)) if row else None


async def select_videos_by_user_id(user_id: int, db: Connection) -> list[Video]:
    cur = await db.execute(SELECT_VIDEOS_BY_USER_ID, {"user_id": user_id})
    rows = await cur.fetchall()
    return [Video.model_validate(dict(row)) for row in rows]


# ==============================================================================
# CORE
# ==============================================================================


async def select_system_name_and_user_translated_name_by_video_id(
    video_id: int, db: Connection
) -> tuple[str | None, str | None]:
    cur = await db.execute(SELECT_SYSTEM_NAME_BY_VIDEO_ID, {"id": video_id})
    row = await cur.fetchone()
    if row is None:
        return None, None
    return row["name"], row["translated_name"]


async def insert_video(video: VideoCreate, db: Connection) -> Video | None:
    await db.execute(INSERT_VIDEO, video.model_dump())
    await db.commit()
    return await select_video_by_aweme_id(aweme_id=video.aweme_id, db=db)


async def upsert_video(video: VideoCreate, db: Connection) -> Video | None:
    await db.execute(UPSERT_VIDEO, video.model_dump())
    await db.commit()
    return await select_video_by_aweme_id(aweme_id=video.aweme_id, db=db)


async def update_video_by_id(
    video_id: int,
    video: VideoUpdate,
    db: Connection,
) -> Video | None:
    existing = await select_video_by_id(video_id=video_id, db=db)
    if not existing:
        return None

    updated = existing.model_copy(update=video.model_dump(exclude_unset=True))
    await db.execute(UPDATE_VIDEO_BY_ID, updated.model_dump())
    await db.commit()
    return updated


async def delete_video_by_id(video_id: int, db: Connection) -> bool:
    existing = await select_video_by_id(video_id=video_id, db=db)
    if not existing:
        return False

    await db.execute(DELETE_VIDEO_BY_ID, {"id": video_id})
    await db.commit()
    return True
