from fastapi import APIRouter, BackgroundTasks, Request

from src.database import DbConnection
from src.job import service as job_service
from src.job.douyin import service as douyin_job_service
from src.job.schemas import JobCreateResponse
from src.tikhub.dependencies import TikHubClientDep

router = APIRouter(prefix="/douyin")


@router.post("/fetch-latest-videos")
async def create_fetch_latest_videos_job(
    request: Request,
    background_tasks: BackgroundTasks,
    db: DbConnection,
    tikhub: TikHubClientDep,
):
    """Create a job that fetches the latest videos for tracked accounts."""
    job = job_service.create_job()
    background_tasks.add_task(
        douyin_job_service.run_fetch_latest_videos_job,
        job.id,
        db,
        tikhub,
    )
    return JobCreateResponse(
        id=job.id,
        events_url=str(request.url_for("get_job_events", job_id=job.id)),
    )


@router.post("/users/{user_id}/fetch-videos")
async def create_fetch_user_videos_job(
    request: Request,
    user_id: int,
    background_tasks: BackgroundTasks,
    db: DbConnection,
    tikhub: TikHubClientDep,
):
    """Create a job that fetches videos for one TikHub user."""
    job = job_service.create_job()
    background_tasks.add_task(
        douyin_job_service.run_fetch_user_videos_job,
        job.id,
        user_id,
        db,
        tikhub,
    )
    return JobCreateResponse(
        id=job.id,
        events_url=str(request.url_for("get_job_events", job_id=job.id)),
    )
