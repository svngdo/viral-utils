from aiosqlite import Connection

from src.douyin import service as douyin_service
from src.job import service as job_service
from src.job.state import jobs
from src.tikhub.client import TikHubClient


async def run_fetch_latest_videos_job(
    job_id: str,
    db: Connection,
    tikhub: TikHubClient,
) -> None:
    """Background job entrypoint for fetching latest Douyin videos."""
    await job_service._run_async_job(
        job_id=job_id,
        events=douyin_service.fetch_latest_videos(
            db=db,
            tikhub=tikhub,
            cancel=jobs[job_id].cancel,
        ),
    )


async def run_fetch_user_videos_job(
    job_id: str,
    user_id: int,
    db: Connection,
    tikhub: TikHubClient,
) -> None:
    """Background job entrypoint for fetching videos from one Douyin user."""
    await job_service._run_async_job(
        job_id=job_id,
        events=douyin_service.fetch_user_videos(
            user_id=user_id,
            db=db,
            tikhub=tikhub,
            cancel=jobs[job_id].cancel,
        ),
    )
