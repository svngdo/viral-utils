import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.config import ensure_app_dirs, settings
from src.database import init_db
from src.douyin.router import router as douyin_router
from src.exceptions import AppException
from src.job.router import router as job_router
from src.logging import setup_logging
from src.platform.router import router as platforms_router
from src.system.router import router as systems_router
from src.video.router import router as video_router

logger = logging.getLogger(__name__)
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_app_dirs(settings)
    await init_db()
    logger.info("Application started")
    yield


# app
app = FastAPI(title="Viral Utils", lifespan=lifespan)


@app.exception_handler(AppException)
async def app_error_handler(request: Request, error: AppException) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={"detail": error.message},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, error: Exception) -> JSONResponse:
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# routes
app.include_router(systems_router)
app.include_router(platforms_router)
app.include_router(douyin_router)
app.include_router(job_router)
app.include_router(video_router)
