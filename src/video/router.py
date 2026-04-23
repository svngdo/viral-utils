from fastapi import APIRouter

from src.config import settings
from src.video import service as video_service
from src.video.config import VideoConfig, reset_config, save_config
from src.video.dependencies import (
    validate_video_dir,
    validate_video_path,
)
from src.video.schemas import (
    InpaintAllRequest,
    InpaintRequest,
)

router = APIRouter(prefix="/video", tags=["video"])


@router.put("/config", response_model=VideoConfig)
def update_config(new_config: VideoConfig):
    save_config(new_config)
    return new_config


@router.post("/config/reset", response_model=VideoConfig)
def reset_config_route():
    reset_config()
    return VideoConfig()


@router.post("/inpaint")
async def inpaint(req: InpaintRequest):
    video_path = validate_video_path(req.video_path)
    video_service.inpaint_video(
        input_path=video_path,
        output_path=settings.processed_dir / video_path.name,
    )


@router.post("/inpaint_all")
async def inpaint_all(req: InpaintAllRequest):
    video_dir = validate_video_dir(req.video_dir)
    video_service.inpaint_all_videos(input_dir=video_dir)
