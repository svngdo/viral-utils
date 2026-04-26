from fastapi import APIRouter, Depends, HTTPException

from src.llm.client import LLMClient
from src.video import service as video_service
from src.video.config import (
    VideoConfig,
    reset_config,
    update_video_config,
)
from src.video.dependencies import get_llm_client
from src.video.exceptions import InpaintingError, TranslationError
from src.video.schemas import (
    InpaintAllResponse,
    InpaintRequest,
    InpaintResponse,
    VideoConfigUpdate,
)

router = APIRouter(prefix="/video", tags=["video"])


@router.put("/config", response_model=VideoConfig)
def update_config(new_config: VideoConfigUpdate):
    updated_config = update_video_config(new_config)
    return updated_config


@router.post("/config/reset", response_model=VideoConfig)
def reset_config_route():
    return reset_config()


@router.post("/inpaint", response_model=InpaintResponse)
async def inpaint(
    req: InpaintRequest,
    llm_client: LLMClient = Depends(get_llm_client),
) -> InpaintResponse:
    try:
        return await video_service.inpaint_video(
            input_path=req.input_path,
            output_path=req.output_path,
            llm_client=llm_client,
        )
    except InpaintingError:
        raise HTTPException(status_code=500, detail="Inpainting failed")
    except TranslationError:
        raise HTTPException(status_code=500, detail="Translation failed")


@router.post("/inpaint_all", response_model=InpaintAllResponse)
async def inpaint_all(
    llm_client: LLMClient = Depends(get_llm_client),
) -> InpaintAllResponse:
    return await video_service.inpaint_all_videos(llm_client=llm_client)
