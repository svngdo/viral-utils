import asyncio

from src.job import service as job_service
from src.job.douyin import service as douyin_job_service
from src.job.schemas import JobStatus, StatusEvent
from src.main import app


def test_job_routes_are_registered():
    routes = [
        route
        for route in app.routes
        if hasattr(route, "methods") and route.path.startswith("/jobs")
    ]
    paths = {route.path for route in routes}

    assert "/jobs/douyin/fetch-latest-videos" in paths
    assert "/jobs/douyin/users/fetch-videos" in paths
    assert "/jobs/douyin/users/{user_id}/fetch-videos" in paths
    assert "/jobs/douyin/download-videos" in paths
    assert "/jobs/douyin/users/download-videos" in paths
    assert "/jobs/video/process-videos" in paths
    assert "/jobs/{job_id}" in paths
    assert "/jobs/{job_id}/cancel" in paths
    assert "/jobs/{job_id}/events" in paths
    assert "/jobs/fetch-latest-videos" not in paths
    assert "/jobs/process-videos" not in paths

    route_paths = [route.path for route in routes]
    assert route_paths.index("/jobs/douyin/users/fetch-videos") < route_paths.index(
        "/jobs/douyin/users/{user_id}/fetch-videos"
    )


def test_legacy_stream_cancel_route_is_not_registered():
    routes = [
        route
        for route in app.routes
        if hasattr(route, "methods") and route.path == "/video/process/stream"
    ]

    assert not any("DELETE" in route.methods for route in routes)


def test_download_videos_job_runner_uses_douyin_download_service(monkeypatch):
    async def run():
        seen_db = object()

        async def download_latest_videos(db, cancel=None):
            assert db is seen_db
            assert cancel is not None
            yield StatusEvent(status=JobStatus.RUNNING)
            yield StatusEvent(status=JobStatus.COMPLETED)

        monkeypatch.setattr(
            "src.job.douyin.service.douyin_service.download_latest_videos",
            download_latest_videos,
        )

        job = job_service.create_job()
        await douyin_job_service.run_download_videos_job(job.id, seen_db)

        assert job.status == JobStatus.COMPLETED

    asyncio.run(run())


def test_download_selected_user_videos_job_runner_uses_douyin_download_service(monkeypatch):
    async def run():
        seen_db = object()
        seen_user_ids = [1, 2]

        async def download_selected_user_videos(user_ids, db, cancel=None):
            assert user_ids == seen_user_ids
            assert db is seen_db
            assert cancel is not None
            yield StatusEvent(status=JobStatus.RUNNING)
            yield StatusEvent(status=JobStatus.COMPLETED)

        monkeypatch.setattr(
            "src.job.douyin.service.douyin_service.download_selected_user_videos",
            download_selected_user_videos,
        )

        job = job_service.create_job()
        await douyin_job_service.run_download_selected_user_videos_job(
            job.id,
            seen_user_ids,
            seen_db,
        )

        assert job.status == JobStatus.COMPLETED

    asyncio.run(run())
