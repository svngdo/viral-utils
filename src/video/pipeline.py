import time
from pathlib import Path

from src.config import settings
from src.llm.client import LLMClient
from src.logging import get_logger
from src.video import cache as video_cache
from src.video import ocr as video_ocr
from src.video.config import VideoConfig
from src.video.ffmpeg import get_video_metadata
from src.video.inpaint import inpaint_and_encode
from src.video.subtitle import merge_subtitles, translate_subtitle, write_srt

logger = get_logger(__name__)


def remove_video_subtitles(
    input_path: str | Path,
    output_path: str | Path,
    srt_path: str | Path,
) -> None:

    # --- INPUT ---
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- DECLARATION ---
    metadata = get_video_metadata(input_path=input_path)
    video_config = VideoConfig()
    client = LLMClient()

    # --- OCR ---
    if video_cache.is_exists(input_path):
        # Read cache
        subtitles = video_cache.read(input_path)
    else:
        ocr_results = video_ocr.ocr(
            input_path=input_path,
            metadata=metadata,
            config=video_config,
        )

        subtitles = merge_subtitles(
            ocr_results=ocr_results,
            metadata=metadata,
            config=video_config,
        )

        # Write cache
        video_cache.write(input_path, subtitles)

    start = time.perf_counter()

    # --- TRANSLATE SUBTITLES ---
    if srt_path.exists():
        logger.info(f"SRT file already exists, skipping: {srt_path.name}")
    else:
        while True:
            try:
                translated_subtitles = translate_subtitle(
                    subtitles=subtitles,
                    llm_client=client,
                    config=video_config,
                )
                write_srt(translated_subtitles, srt_path=srt_path)
                break
            except Exception:
                pass

    logger.info(f"Translate time {time.perf_counter() - start}")

    # --- INPAINT & ENCODE ---
    inpaint_and_encode(
        input_path=input_path,
        output_path=output_path,
        metadata=metadata,
        subtitles=subtitles,
        config=video_config,
    )


def remove_video_subtitles_by_dir(in_dir: str | Path, out_dir: str | Path):
    start = time.perf_counter()

    in_dir = Path(in_dir).expanduser()
    out_dir = Path(out_dir).expanduser()
    files = list(in_dir.glob("*.mp4"))
    processed = 0

    for f in files:
        output_path = out_dir / f.name
        srt_path = out_dir / f"{f.stem}.srt"

        if output_path.exists():
            logger.info(f"Video already processed, skipping: {f.name}")
            continue

        remove_video_subtitles(
            input_path=f,
            output_path=output_path,
            srt_path=srt_path,
        )

        f.rename(settings.archives_raw_dir / f.name)

        processed += 1

    elapsed = time.perf_counter() - start
    logger.info(f"Processed {processed} files in {elapsed}")
