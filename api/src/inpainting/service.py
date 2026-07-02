import logging
import threading
import time
from collections.abc import Generator
from pathlib import Path

from src.inpainting.engine import InpaintEngineProtocol
from src.inpainting.schemas import InpaintConfig
from src.job import service as job_service
from src.job.exceptions import JobCancelled
from src.job.schemas import (
    LogEvent,
    SSEEvent,
)
from src.subtitle.schemas import Subtitle
from src.video.engine import VideoEngineProtocol

logger = logging.getLogger(__name__)


def inpaint(
    video_path: str | Path,
    out_path: str | Path,
    subtitles: list[Subtitle],
    video_engine: VideoEngineProtocol,
    engine: InpaintEngineProtocol,
    config: InpaintConfig,
    cancel: threading.Event | None = None,
) -> Generator[SSEEvent]:
    video_path = Path(video_path)
    out_path = Path(out_path)

    job_service.raise_if_cancelled(cancel)

    if out_path.exists():
        yield LogEvent(message=f"File already processed: {out_path}")
        return

    meta = video_engine.get_metadata(video_path)
    subtitles = [s for s in subtitles if s.conf >= config.conf_threshold]

    if not subtitles:
        job_service.raise_if_cancelled(cancel)
        video_engine.copy(video_path, out_path)
        yield LogEvent(message="No subtitles to inpaint")
        return

    job_service.raise_if_cancelled(cancel)
    encoder = video_engine.get_encoder(path=video_path, out_path=out_path)

    if encoder.stdin is None:
        raise BrokenPipeError("ffmpeg encoder failed to open stdin pipe")
    try:
        for frame in video_engine.iter_frames(video_path):
            try:
                job_service.raise_if_cancelled(cancel)
            except JobCancelled:
                encoder.kill()
                raise

            active = [s for s in subtitles if s.start <= frame.timestamp <= s.end]
            data = frame.data

            if active:
                job_service.raise_if_cancelled(cancel)
                data = engine.inpaint(frame.data, bboxes=[s.bbox for s in active])

            job_service.raise_if_cancelled(cancel)
            encoder.stdin.write(data.tobytes())

            if frame.index % 10 == 0:
                yield LogEvent(
                    message=f"Inpainting: {frame.index}/{meta.total_frames} frames"
                )

            # Throttle to keep device cool
            time.sleep(config.delay)

    finally:
        if encoder.stdin:
            encoder.stdin.close()
        encoder.wait()
        if encoder.returncode != 0 and not (cancel and cancel.is_set()):
            err = (
                encoder.stderr.read().decode(errors="replace") if encoder.stderr else ""
            )
            raise RuntimeError(f"ffmpeg encode failed: {err}")

    yield LogEvent(message=f"Saved processed video: {out_path}")
    yield LogEvent(message="Inpainting completed")
