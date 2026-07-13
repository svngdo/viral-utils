from src.inpainting.engine import InpaintEngineProtocol
from src.inpainting.schemas import InpaintConfig
from src.job import service as job_service
from src.job.state import jobs
from src.llm.schemas import LLMModel
from src.ocr.engine import OcrEngine
from src.ocr.schemas import OcrConfig
from src.subtitle.schemas import SubtitleConfig
from src.video import service as video_service
from src.video.engine import VideoEngineProtocol
from src.video.schemas import ProcessVideosRequest


async def run_process_videos_job(
    job_id: str,
    request: ProcessVideosRequest,
    video_engine: VideoEngineProtocol,
    ocr_engine: OcrEngine,
    ocr_config: OcrConfig,
    subtitle_config: SubtitleConfig,
    inpaint_engine: InpaintEngineProtocol,
    inpaint_config: InpaintConfig,
    llm_models: list[LLMModel],
) -> None:
    """Background job entrypoint for the sync video processing pipeline."""
    state = jobs[job_id]
    await job_service._run_sync_job(
        job_id=job_id,
        events=video_service.process(
            request=request,
            video_engine=video_engine,
            ocr_engine=ocr_engine,
            ocr_config=ocr_config,
            subtitle_config=subtitle_config,
            inpaint_engine=inpaint_engine,
            inpaint_config=inpaint_config,
            llm_models=llm_models,
            cancel=state.cancel,
        ),
    )
