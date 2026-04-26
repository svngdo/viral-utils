import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from send2trash import send2trash

from src.llm.client import LLMClient
from src.logging import get_logger
from src.video import ffmpeg, ocr, renderer, subtitle
from src.video.config import video_config
from src.video.constants import ALLOWED_EXTENSIONS
from src.video.exceptions import InpaintingError, TranslationError
from src.video.schemas import (
    InpaintAllResponse,
    InpaintResponse,
    InpaintResult,
    InpaintStatus,
)

logger = get_logger(__name__)


async def inpaint_video(
    input_path: str | Path,
    output_path: str | Path,
    llm_client: LLMClient,
) -> InpaintResponse:
    """Inpaint hardcoded subtitles from a video frame by frame.

    Args:
        input_path: Path to the source video file.
        output_path: Path to save the inpainted video.

    Returns:
        Path to the inpainted video file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path = output_path.parent / f"{input_path.stem}.srt"
    metadata = ffmpeg.get_video_metadata(input_path=input_path)

    try:
        # ocr
        ocr_subs = ocr.run(
            video_path=input_path,
            metadata=metadata,
        )
        # merge subtitles
        merged_subs = subtitle.merge_subtitles(
            subtitles=ocr_subs,
            metadata=metadata,
        )

        translate_task = asyncio.create_task(
            subtitle.translate_subtitle(
                subtitles=merged_subs,
                llm_client=llm_client,
            )
        )

        # inpaint subtitles
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                renderer.inpaint_video,
                input_path,
                metadata,
                merged_subs,
                output_path,
            )

        translated_subs = await translate_task
        subtitle.write_srt(subtitles=translated_subs, srt_path=srt_path)
    except TranslationError as e:
        logger.exception("Translation failed - %s - %s", input_path.name, e)
        raise
    except InpaintingError as e:
        logger.exception("Inpainting failed - %s - %s", input_path.name, e)
        raise

    return InpaintResponse(output_path=str(output_path), srt_path=str(srt_path))


async def inpaint_all_videos(llm_client: LLMClient) -> InpaintAllResponse:
    input_dir = Path(video_config.raw_dir)
    output_dir = Path(video_config.processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[InpaintResult] = []

    files = input_dir.glob("*")
    files = [f for f in files if f.suffix in ALLOWED_EXTENSIONS]
    for f in files:
        try:
            await inpaint_video(
                input_path=f,
                output_path=output_dir / f.name,
                llm_client=llm_client,
            )
            send2trash(str(f))
            results.append(
                InpaintResult(
                    file=f.name,
                    status=InpaintStatus.OK,
                )
            )
        except (InpaintingError, TranslationError) as e:
            logger.exception("Failed to process %s", f.name)
            results.append(
                InpaintResult(
                    file=f.name,
                    status=InpaintStatus.ERROR,
                    error=str(e),
                )
            )

    return InpaintAllResponse(
        total=len(files),
        succeeded=sum(1 for r in results if r.status == InpaintStatus.OK),
        failed=sum(1 for r in results if r.status == InpaintStatus.ERROR),
        results=results,
    )
