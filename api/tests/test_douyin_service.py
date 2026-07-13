import asyncio

import pytest

from src.douyin.schemas import User, UserStatus, Video
from src.douyin.service import (
    _sync_user_videos,
    fetch_selected_user_videos,
    fetch_user_latest_videos,
)
from src.tikhub.exceptions import TikHubRequestError, TikHubStatusError


class FailingTikHubClient:
    def __init__(self):
        self.calls = 0

    async def fetch_user_post_videos(self, sec_uid: str):
        self.calls += 1
        raise TikHubRequestError()


class BadRequestTikHubClient:
    def __init__(self):
        self.calls = 0

    async def fetch_user_post_videos(self, sec_uid: str):
        self.calls += 1
        raise TikHubStatusError(upstream_status_code=400)


def test_fetch_user_latest_videos_returns_empty_after_tikhub_retries():
    async def run():
        client = FailingTikHubClient()

        user, videos = await fetch_user_latest_videos("sec-uid", client)

        assert client.calls == 3
        assert user == {}
        assert videos == []

    asyncio.run(run())


def test_fetch_user_latest_videos_does_not_retry_bad_request():
    async def run():
        client = BadRequestTikHubClient()

        user, videos = await fetch_user_latest_videos("sec-uid", client)

        assert client.calls == 1
        assert user == {}
        assert videos == []

    asyncio.run(run())


@pytest.mark.anyio
async def test_fetch_selected_user_videos_emits_progress_per_user(monkeypatch):
    users = {
        1: User(
            id=1,
            sec_uid="sec-1",
            name="User 1",
            status=UserStatus.ACTIVE,
            created_at=1,
            updated_at=1,
        ),
        2: User(
            id=2,
            sec_uid="sec-2",
            name="User 2",
            status=UserStatus.ACTIVE,
            created_at=1,
            updated_at=1,
        ),
    }
    synced_user_ids = []

    async def select_user_by_id(user_id, db):
        return users[user_id]

    async def fetch_latest(sec_uid, tikhub):
        return {"name": f"Fetched {sec_uid}"}, []

    async def sync_user(existing, fetched_user, db, cancel=None):
        synced_user_ids.append(existing.id)
        return existing

    async def sync_user_videos(existing_user, fetched_videos, db, cancel=None):
        return None

    monkeypatch.setattr("src.douyin.service.repo.select_user_by_id", select_user_by_id)
    monkeypatch.setattr("src.douyin.service.fetch_user_latest_videos", fetch_latest)
    monkeypatch.setattr("src.douyin.service._sync_user", sync_user)
    monkeypatch.setattr("src.douyin.service._sync_user_videos", sync_user_videos)

    events = [
        event
        async for event in fetch_selected_user_videos(
            user_ids=[1, 2],
            db=None,
            tikhub=None,
        )
    ]

    assert synced_user_ids == [1, 2]
    progress = [event for event in events if event.type == "progress"]
    assert [(event.done, event.total) for event in progress] == [(1, 2), (2, 2)]
    assert events[0].type == "status"
    assert events[0].status == "running"
    assert events[-1].type == "status"
    assert events[-1].status == "completed"


@pytest.mark.anyio
async def test_fetch_selected_user_videos_reports_empty_tikhub_response(monkeypatch):
    user = User(
        id=1,
        sec_uid="sec-1",
        name="User 1",
        status=UserStatus.ACTIVE,
        created_at=1,
        updated_at=1,
    )

    async def select_user_by_id(user_id, db):
        return user

    async def fetch_latest(sec_uid, tikhub):
        return {}, []

    monkeypatch.setattr("src.douyin.service.repo.select_user_by_id", select_user_by_id)
    monkeypatch.setattr("src.douyin.service.fetch_user_latest_videos", fetch_latest)

    events = [
        event
        async for event in fetch_selected_user_videos(
            user_ids=[1],
            db=None,
            tikhub=None,
        )
    ]

    logs = [event.message for event in events if event.type == "log"]
    assert "Skipped User 1: TikHub returned no videos for sec_uid=sec-1" in logs
    progress = [event for event in events if event.type == "progress"]
    assert [(event.done, event.total) for event in progress] == [(1, 1)]
    assert events[-1].type == "status"
    assert events[-1].status == "completed"


@pytest.mark.anyio
async def test_sync_user_videos_preserves_download_state_for_existing_video(
    monkeypatch,
):
    user = User(
        id=1,
        sec_uid="sec-1",
        name="User 1",
        status=UserStatus.ACTIVE,
        created_at=1,
        updated_at=1,
    )
    existing_video = Video(
        id=10,
        aweme_id="video-1",
        title="same title",
        translated_title="same title translated",
        create_time=1,
        digg_count=1,
        duration=10,
        urls='["old"]',
        is_downloaded=True,
        user_id=user.id,
        created_at=1,
        updated_at=1,
    )
    updates = []

    async def select_videos_by_user_id(user_id, db):
        return [existing_video]

    async def update_video_by_id(video_id, video, db):
        updates.append((video_id, video))
        return existing_video.model_copy(update=video.model_dump(exclude_unset=True))

    monkeypatch.setattr(
        "src.douyin.service.repo.select_videos_by_user_id",
        select_videos_by_user_id,
    )
    monkeypatch.setattr(
        "src.douyin.service.repo.update_video_by_id",
        update_video_by_id,
    )

    await _sync_user_videos(
        existing_user=user,
        fetched_videos=[
            {
                "aweme_id": "video-1",
                "title": "same title",
                "digg_count": 99,
                "duration": 20,
                "urls": '["new"]',
                "is_downloaded": False,
            }
        ],
        db=None,
    )

    assert len(updates) == 1
    video_id, update = updates[0]
    assert video_id == existing_video.id
    assert update.digg_count == 99
    assert update.duration == 20
    assert update.urls == '["new"]'
    assert update.is_downloaded is None
