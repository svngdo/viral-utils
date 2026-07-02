import logging
import threading
from collections.abc import Generator
from pathlib import Path

from src.inpainting import service as inpaint_service
from src.inpainting.engine import InpaintEngineProtocol
from src.inpainting.schemas import InpaintConfig
from src.job import service as job_service
from src.job.exceptions import JobCancelled
from src.job.schemas import (
    JobStatus,
    LogEvent,
    ProgressEvent,
    SSEEvent,
    StatusEvent,
)
from src.llm.schemas import LLMModel
from src.ocr import service as ocr_service
from src.ocr.engine import OcrEngine
from src.ocr.schemas import OcrConfig
from src.subtitle import service as subtitle_service
from src.subtitle.schemas import SubtitleConfig
from src.video.constants import ALLOWED_EXTENSIONS
from src.video.engine import VideoEngineProtocol
from src.video.schemas import ProcessVideosRequest

logger = logging.getLogger(__name__)


def _child_event_to_log(event: SSEEvent, phase: str) -> LogEvent | StatusEvent | None:
    if isinstance(event, StatusEvent):
        return None
    if isinstance(event, ProgressEvent):
        return LogEvent(message=f"{phase}: {event.done}/{event.total} frames")
    return event


def process(
    request: ProcessVideosRequest,
    video_engine: VideoEngineProtocol,
    ocr_engine: OcrEngine,
    ocr_config: OcrConfig,
    subtitle_config: SubtitleConfig,
    inpaint_engine: InpaintEngineProtocol,
    inpaint_config: InpaintConfig,
    llm_models: list[LLMModel],
    cancel: threading.Event | None = None,
) -> Generator[SSEEvent]:
    in_dir = Path(request.in_dir)
    out_dir = Path(request.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(
        f for f in in_dir.glob("*") if f.suffix.lower() in ALLOWED_EXTENSIONS
    )
    if not videos:
        yield LogEvent(message="No videos to process")
        yield StatusEvent(status=JobStatus.COMPLETED)
        return

    total = len(videos)
    yield StatusEvent(status=JobStatus.RUNNING)
    yield LogEvent(message=f"Processing {total} videos")
    yield ProgressEvent(done=0, total=total)

    completed = 0

    for video in videos:
        job_service.raise_if_cancelled(cancel)

        yield LogEvent(message=f"Started {video.name}")

        # --- OCR ---
        ocr_generator = ocr_service.extract(
            video_path=video,
            video_engine=video_engine,
            engine=ocr_engine,
            config=ocr_config,
            cancel=cancel,
        )
        try:
            while True:
                ocr_event = next(ocr_generator)
                yield ocr_event
        except JobCancelled:
            yield LogEvent(message="Processing cancelled during OCR")
            raise
        except StopIteration as e:
            ocr_subs = e.value

        if not ocr_subs:
            yield LogEvent(message=f"No subtitles found in {video.name}")
            out_path = out_dir / video.name
            video_engine.copy(video, out_path)
            yield LogEvent(message=f"Saved processed video: {out_path}")
            completed += 1
            yield LogEvent(message=f"Finished {video.name}")
            yield ProgressEvent(done=completed, total=total)
            continue

        # --- Merge subtitles ---
        job_service.raise_if_cancelled(cancel)

        merged_subs = subtitle_service.merge(
            video_path=video,
            video_engine=video_engine,
            subtitles=ocr_subs,
            subtitle_config=subtitle_config,
        )
        yield LogEvent(message="Merged subtitles")

        # --- Translate merged subtitles ---
        job_service.raise_if_cancelled(cancel)

        srt_path = out_dir / f"{video.stem}.srt"
        translate_generator = subtitle_service.translate(
            video_path=video,
            subtitles=merged_subs,
            llm_models=llm_models,
        )
        try:
            while True:
                translate_event = next(translate_generator)
                yield translate_event
        except StopIteration as e:
            translated_subs = e.value
        job_service.raise_if_cancelled(cancel)

        subtitle_service.write_srt(subtitles=translated_subs, srt_path=srt_path)
        yield LogEvent(message=f"Saved subtitles: {srt_path.name}")

        job_service.raise_if_cancelled(cancel)

        # --- Inpaint ---
        try:
            yield from inpaint_service.inpaint(
                video_path=video,
                out_path=out_dir / video.name,
                video_engine=video_engine,
                engine=inpaint_engine,
                subtitles=translated_subs,
                config=inpaint_config,
                cancel=cancel,
            )
        except JobCancelled:
            yield LogEvent(message="Processing cancelled during inpainting")
            raise

        # --- Move video to trash after process ---
        yield LogEvent(message=f"Finished {video.name}")
        completed += 1
        yield ProgressEvent(done=completed, total=total)

    yield LogEvent(message=f"Completed processing {completed}/{total} videos")
    yield StatusEvent(status=JobStatus.COMPLETED)
