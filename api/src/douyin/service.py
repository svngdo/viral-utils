import asyncio
import json
import logging
import threading
from collections.abc import AsyncGenerator, Awaitable
from datetime import datetime
from pathlib import Path
from typing import Any

from aiosqlite import Connection
from fastapi.concurrency import run_in_threadpool

from src.config import settings
from src.douyin import repository as repo
from src.douyin.exceptions import (
    FetchUserVideosError,
    InsertUserError,
    UserExistsError,
    UserNotFoundError,
)
from src.douyin.schemas import (
    User,
    UserCreate,
    UserResponse,
    UserUpdate,
    Video,
    VideoCreate,
    VideoUpdate,
)
from src.download import service as download_service
from src.job import service as job_service
from src.job.schemas import (
    JobStatus,
    LogEvent,
    ProgressEvent,
    SSEEvent,
    StatusEvent,
)
from src.tikhub.client import TikHubClient
from src.tikhub.exceptions import TikHubError, TikHubStatusError
from src.tikhub.schemas import AwemeItem
from src.translation import service as translate_service

logger = logging.getLogger(__name__)


# ==============================================================================
# CRUD
# ==============================================================================


async def get_user_by_sec_uid(sec_uid: str, db: Connection) -> UserResponse:
    db_user = await repo.select_user_by_sec_uid(sec_uid=sec_uid, db=db)
    if not db_user:
        raise UserNotFoundError()
    return UserResponse.model_validate(db_user)


async def get_users(db: Connection) -> list[UserResponse]:
    db_users = await repo.select_users(db)
    return [UserResponse.model_validate(user) for user in db_users]


# ==============================================================================
# HELPERS
# ==============================================================================


async def _gather_with_cancel(
    awaitables: list[Awaitable[Any]],
    cancel: threading.Event | None = None,
) -> list[Any]:
    task = asyncio.gather(*awaitables)
    try:
        while not task.done():
            job_service.raise_if_cancelled(cancel)
            await asyncio.sleep(0.1)

        return list(await task)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def _get_video_urls(video_data: AwemeItem) -> list[str]:
    video = video_data.video
    if video.play_addr_265:
        urls = video.play_addr_265.url_list
    elif video.play_addr_h264:
        urls = video.play_addr_h264.url_list
    elif video.play_addr:
        urls = video.play_addr.url_list
    elif video.download_addr:
        urls = video.download_addr.url_list
    else:
        urls = []
    return urls


async def _translate_text(text: str | None) -> str | None:
    if not text:
        return None
    try:
        return await run_in_threadpool(translate_service.translate, text)
    except Exception:
        logger.warning("Translation failed, using original - text=%s", text)
        return text


async def _translate_video_titles(
    videos: list[dict[str, Any]],
) -> list[str | None]:
    titles = await asyncio.gather(*[_translate_text(video["title"]) for video in videos])
    return list(titles)


def _display_user_name(user: User) -> str:
    return user.translated_name or user.name or user.sec_uid


def _display_video_title(video: Video) -> str:
    return video.translated_title or video.title or video.aweme_id


# ==============================================================================
# CORE
# ==============================================================================


async def _insert_user(
    user: UserCreate,
    fetched_user: dict,
    db: Connection,
) -> User:
    name = fetched_user["name"]
    translated_name = await _translate_text(name)
    saved = await repo.insert_user(
        user=user.model_copy(
            update={
                "name": name,
                "translated_name": translated_name,
                "last_fetched": int(datetime.now().timestamp()),
            }
        ),
        db=db,
    )
    if not saved:
        raise InsertUserError()
    return saved


async def _insert_videos(
    videos: list[dict[str, Any]],
    translated_titles: list[str | None],
    user_id: int,
    db: Connection,
) -> AsyncGenerator[SSEEvent]:
    total = len(videos)
    for i, (video, translated_title) in enumerate(
        zip(videos, translated_titles, strict=True), start=1
    ):
        try:
            await repo.upsert_video(
                video=VideoCreate(
                    **{
                        **video,
                        "translated_title": translated_title,
                        "user_id": user_id,
                    }
                ),
                db=db,
            )
        except Exception as e:
            logger.error(
                "Failed to upsert video - aweme_id=%s - %s", video["aweme_id"], e
            )
            yield LogEvent(message=f"Failed to save video {i}/{total}")
            continue
        yield LogEvent(message=f"Saved video {i}/{total}")
        yield ProgressEvent(done=i, total=total)


async def fetch_user_latest_videos(
    sec_uid: str,
    tikhub: TikHubClient,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    response = None
    for attempt in range(1, 4):
        try:
            response = await tikhub.fetch_user_post_videos(sec_uid=sec_uid)
            break
        except TikHubStatusError as e:
            if e.upstream_status_code and 400 <= e.upstream_status_code < 500:
                logger.warning(
                    "TikHub rejected user fetch - sec_uid=%s - %s", sec_uid, e
                )
                break
            logger.exception("Failed to fetch user - attempt=%d", attempt)
            continue
        except TikHubError:
            logger.exception("Failed to fetch user - attempt=%d", attempt)
            continue
        except Exception:
            raise

    if not response or not response.data or not response.data.aweme_list:
        return {}, []

    aweme_items = response.data.aweme_list
    name = aweme_items[0].author.nickname
    user = {"sec_uid": sec_uid, "name": name}
    aweme_items = [
        {
            "aweme_id": item.aweme_id,
            "title": item.desc or item.caption,
            "create_time": item.create_time,
            "digg_count": item.statistics.digg_count,
            "duration": item.duration,
            "urls": json.dumps(_get_video_urls(video_data=item), ensure_ascii=False),
            "is_downloaded": False,
        }
        for item in response.data.aweme_list
    ]

    return user, aweme_items


async def create_user_videos(
    user: UserCreate,
    db: Connection,
    tikhub: TikHubClient,
) -> AsyncGenerator[SSEEvent]:
    yield StatusEvent(status=JobStatus.RUNNING)
    yield LogEvent(message="Fetching user")

    fetched_user, fetched_videos = await fetch_user_latest_videos(
        sec_uid=user.sec_uid,
        tikhub=tikhub,
    )
    if not fetched_user:
        raise FetchUserVideosError()

    existing = await repo.select_user_by_sec_uid(sec_uid=user.sec_uid, db=db)
    if existing:
        raise UserExistsError()

    yield LogEvent(message="Saving user")
    saved = await _insert_user(user=user, fetched_user=fetched_user, db=db)

    yield LogEvent(message="Translating video titles")
    translated_titles = await _translate_video_titles(fetched_videos)

    yield LogEvent(message="Saving videos")
    async for event in _insert_videos(
        videos=fetched_videos,
        translated_titles=translated_titles,
        user_id=saved.id,
        db=db,
    ):
        yield event

    yield LogEvent(message="Done")
    yield StatusEvent(status=JobStatus.COMPLETED)


async def _sync_user(
    existing: User,
    fetched_user: dict,
    db: Connection,
    cancel: threading.Event | None = None,
) -> User:
    name = fetched_user["name"]
    translated_name = existing.translated_name
    if name and name != existing.name:
        job_service.raise_if_cancelled(cancel)
        translated_name = await _translate_text(name)
        job_service.raise_if_cancelled(cancel)

    job_service.raise_if_cancelled(cancel)
    updated = await repo.update_user_by_id(
        user_id=existing.id,
        user=UserUpdate(
            name=name or existing.name,
            translated_name=translated_name,
            last_fetched=int(datetime.now().timestamp()),
        ),
        db=db,
    )
    return updated or existing


async def _sync_user_videos(
    existing_user: User,
    fetched_videos: list[dict[str, Any]],
    db: Connection,
    cancel: threading.Event | None = None,
) -> None:
    job_service.raise_if_cancelled(cancel)

    db_videos = await repo.select_videos_by_user_id(user_id=existing_user.id, db=db)
    db_videos_by_aweme_id = {v.aweme_id: v for v in db_videos}

    new_videos = []
    update_videos = []

    for video in fetched_videos:
        existing = db_videos_by_aweme_id.get(video["aweme_id"])
        if not existing:
            new_videos.append(video)
        else:
            update_videos.append((existing, video))

    # translate new videos in batch
    job_service.raise_if_cancelled(cancel)
    translated_titles = await _translate_video_titles(new_videos)
    job_service.raise_if_cancelled(cancel)

    for video, translated_title in zip(new_videos, translated_titles, strict=True):
        job_service.raise_if_cancelled(cancel)
        try:
            await repo.insert_video(
                video=VideoCreate(
                    **{
                        **video,
                        "translated_title": translated_title,
                        "user_id": existing_user.id,
                    }
                ),
                db=db,
            )
        except Exception as e:
            logger.error(
                "Failed to insert video - aweme_id=%s - %s",
                video["aweme_id"],
                e,
            )

    for existing_video, video in update_videos:
        job_service.raise_if_cancelled(cancel)
        title = video["title"]
        translated_title = existing_video.translated_title
        if title and title != existing_video.title:
            job_service.raise_if_cancelled(cancel)
            translated_title = await _translate_text(title)
            job_service.raise_if_cancelled(cancel)
        try:
            await repo.update_video_by_id(
                video_id=existing_video.id,
                video=VideoUpdate(
                    title=title,
                    translated_title=translated_title,
                    digg_count=video["digg_count"],
                    duration=video["duration"],
                    urls=video["urls"],
                    is_downloaded=video["is_downloaded"],
                ),
                db=db,
            )
        except Exception as e:
            logger.error(
                "Failed to update video - aweme_id=%s - %s", video["aweme_id"], e
            )


async def fetch_latest_videos(
    db: Connection,
    tikhub: TikHubClient,
    cancel: threading.Event | None = None,
) -> AsyncGenerator[SSEEvent]:
    job_service.raise_if_cancelled(cancel)

    active_users = await repo.select_users_to_fetch(db=db)
    active_users = active_users[:3]
    total = len(active_users)

    yield StatusEvent(status=JobStatus.RUNNING)
    yield LogEvent(message="Fetching latest videos")

    job_service.raise_if_cancelled(cancel)

    results = await _gather_with_cancel(
        [
            fetch_user_latest_videos(sec_uid=u.sec_uid, tikhub=tikhub)
            for u in active_users
        ],
        cancel=cancel,
    )

    job_service.raise_if_cancelled(cancel)

    for i, (user, (fetched_user, fetched_videos)) in enumerate(
        zip(active_users, results, strict=True), start=1
    ):
        job_service.raise_if_cancelled(cancel)

        if not fetched_user:
            logger.warning("Skip - failed to fetch - sec_uid=%s", user.sec_uid)
            continue

        yield LogEvent(message=f"Syncing {_display_user_name(user)}")
        yield ProgressEvent(done=i, total=total)

        existing = await repo.select_user_by_sec_uid(sec_uid=user.sec_uid, db=db)
        if not existing:
            logger.warning("Skip - user not found - sec_uid=%s", user.sec_uid)
            continue

        job_service.raise_if_cancelled(cancel)

        await _sync_user(
            existing=existing,
            fetched_user=fetched_user,
            db=db,
            cancel=cancel,
        )
        job_service.raise_if_cancelled(cancel)

        await _sync_user_videos(
            existing_user=existing,
            fetched_videos=fetched_videos,
            db=db,
            cancel=cancel,
        )

    yield LogEvent(message="Done")
    yield StatusEvent(status=JobStatus.COMPLETED)


async def fetch_user_videos(
    user_id: int,
    db: Connection,
    tikhub: TikHubClient,
    cancel: threading.Event | None = None,
) -> AsyncGenerator[SSEEvent]:
    job_service.raise_if_cancelled(cancel)

    existing = await repo.select_user_by_id(user_id=user_id, db=db)
    if not existing:
        raise UserNotFoundError()

    yield LogEvent(message=f"Fetching {_display_user_name(existing)}")
    yield StatusEvent(status=JobStatus.RUNNING)

    job_service.raise_if_cancelled(cancel)

    fetched_user, fetched_videos = await fetch_user_latest_videos(
        sec_uid=existing.sec_uid,
        tikhub=tikhub,
    )
    if not fetched_user:
        raise FetchUserVideosError()

    job_service.raise_if_cancelled(cancel)

    yield LogEvent(message="Syncing user")
    synced = await _sync_user(
        existing=existing,
        fetched_user=fetched_user,
        db=db,
        cancel=cancel,
    )

    job_service.raise_if_cancelled(cancel)

    yield LogEvent(message="Syncing videos")
    await _sync_user_videos(
        existing_user=synced,
        fetched_videos=fetched_videos,
        db=db,
        cancel=cancel,
    )

    job_service.raise_if_cancelled(cancel)

    yield LogEvent(message="Done")
    yield StatusEvent(status=JobStatus.COMPLETED)


async def _get_video_download_path(video: Video, db: Connection) -> Path:
    dt = datetime.fromtimestamp(video.create_time).strftime("%Y%m%d")
    title_prefix = f"{dt}-{video.digg_count}-{video.aweme_id}"

    (
        system_name,
        user_translated_name,
    ) = await repo.select_system_name_and_user_translated_name_by_video_id(
        video_id=video.id,
        db=db,
    )

    # build path
    if system_name and user_translated_name:
        path = settings.source_dir / system_name / user_translated_name
    elif system_name:
        path = settings.source_dir / system_name
    elif user_translated_name:
        path = settings.source_dir / user_translated_name
    else:
        path = settings.source_dir

    # strip hashtags from title
    stem = "untitled"
    if video.translated_title:
        stem = video.translated_title.split("#")[0].strip() if video.title else ""
    title = f"{title_prefix}-{stem}"
    title = f"{title[: settings.file_name_max_len]}.mp4"

    return path / title


async def download_video(video: Video, db: Connection) -> tuple[Video, bool]:
    if not video.urls:
        logger.warning("No URLs found - aweme_id=%s", video.aweme_id)
        return video, False

    path = await _get_video_download_path(video=video, db=db)
    if path.exists():
        logger.warning("Video already exists - aweme_id=%s", video.aweme_id)
        return video, True

    path.parent.mkdir(parents=True, exist_ok=True)
    for url in json.loads(video.urls):
        for attempt in range(3):
            try:
                download_ok = await asyncio.to_thread(
                    download_service.download, url, path
                )
                if download_ok:
                    return video, True
            except Exception:
                logger.error(
                    "Error during download - attempt=%d - aweme_id=%d",
                    attempt + 1,
                    video.aweme_id,
                )
            logger.error(
                "Failed to download video - attempt=%d - aweme_id=%s - url=%s",
                attempt + 1,
                video.aweme_id,
                url,
            )
    return video, False


async def download_latest_videos(db: Connection) -> AsyncGenerator[SSEEvent]:
    yield StatusEvent(status=JobStatus.RUNNING)
    yield LogEvent(message="Downloading latest videos")

    available_videos = await repo.select_videos_to_download(db=db)
    if not available_videos:
        yield LogEvent(message="No videos to download")
        yield StatusEvent(status=JobStatus.COMPLETED)
        return

    total = len(available_videos)
    completed = 0

    tasks = [
        asyncio.create_task(
            download_video(video=v, db=db),
            name=f"download-{v.aweme_id}",
        )
        for v in available_videos
    ]

    for coro in asyncio.as_completed(tasks):
        video, result = await coro

        completed += 1
        if video.is_downloaded == result:
            continue

        await repo.update_video_by_id(
            video_id=video.id,
            video=VideoUpdate(is_downloaded=result),
            db=db,
        )

        yield LogEvent(
            message=f"{'Downloaded' if result else 'Failed'}: {_display_video_title(video)}"
        )
        yield ProgressEvent(done=completed, total=total)

    yield LogEvent(message=f"{completed}/{total} videos are downloaded")
    yield StatusEvent(status=JobStatus.COMPLETED)
