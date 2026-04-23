from pathlib import Path

from src.logging import get_logger
from src.video import ffmpeg, ocr, renderer, subtitle
from src.video.config import video_config
from src.video.constants import ALLOWED_EXTENSIONS

logger = get_logger(__name__)


def inpaint_video(input_path: str | Path, output_path: str | Path) -> None:
    """Inpaint hardcoded subtitles from a video frame by frame.

    Args:
        input_path: Path to the source video file.
        output_path: Path to save the inpainted video.

    Returns:
        Path to the inpainted video file.
    """
    metadata = ffmpeg.get_video_metadata(input_path=input_path)

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
    # inpaint subtitles
    renderer.inpaint_video(
        input_path=input_path,
        metadata=metadata,
        subtitles=merged_subs,
        output_path=output_path,
    )


def inpaint_all_videos(input_dir: str | Path) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(video_config.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    files = input_dir.glob("*")
    files = [f for f in files if f.suffix in ALLOWED_EXTENSIONS]

    for f in files:
        inpaint_video(input_path=f, output_path=output_dir / f.name)
