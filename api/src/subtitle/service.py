import json
import logging
import time
from collections.abc import Generator
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.config import get_settings
from src.job.schemas import LogEvent, SSEEvent
from src.llm import service as llm_service
from src.llm.schemas import LLMModel
from src.subtitle.constants import TRANSLATE_SUBTITLES
from src.subtitle.schemas import BoundingBox, Subtitle, SubtitleConfig
from src.video.engine import VideoEngineProtocol, VideoMetadata

logger = logging.getLogger(__name__)


def _update_subtitle(subtitle: Subtitle, result: Subtitle) -> None:
    subtitle.end = max(subtitle.end, result.end)
    subtitle.bbox.x1 = min(subtitle.bbox.x1, result.bbox.x1)
    subtitle.bbox.y1 = min(subtitle.bbox.y1, result.bbox.y1)
    subtitle.bbox.x2 = max(subtitle.bbox.x2, result.bbox.x2)
    subtitle.bbox.y2 = max(subtitle.bbox.y2, result.bbox.y2)

    if result.conf > subtitle.conf:
        subtitle.text = result.text
        subtitle.conf = result.conf


def _pad_subtitles(
    subtitles: list[Subtitle], meta: VideoMetadata, sub_config: SubtitleConfig
) -> list[Subtitle]:
    frame_duration = 1 / meta.fps
    padding = frame_duration * sub_config.frame_padding
    return [
        s.model_copy(
            update={
                "start": max(s.start - padding, 0),
                "end": min(s.end + padding, meta.duration),
            }
        )
        for s in subtitles
    ]


def _is_same_box(
    box1: BoundingBox,
    box2: BoundingBox,
    iou_threshold: float,
) -> bool:
    """Check if two bounding boxes overlap sufficiently (IoU-based)."""
    x_left = max(box1.x1, box2.x1)
    y_bottom = max(box1.y1, box2.y1)
    x_right = min(box1.x2, box2.x2)
    y_top = min(box1.y2, box2.y2)

    if x_right <= x_left or y_top <= y_bottom:
        return False

    intersection = (x_right - x_left) * (y_top - y_bottom)
    area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
    area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
    union = area1 + area2 - intersection

    return (intersection / union) >= iou_threshold if union > 0 else False


# TODO: update the merge algorithm
def merge(
    video_path: str | Path,
    video_engine: VideoEngineProtocol,
    subtitles: list[Subtitle],
    subtitle_config: SubtitleConfig,
) -> list[Subtitle]:
    """Merge related subtitles into 1 segment."""

    if not subtitles:
        return []

    meta = video_engine.get_metadata(video_path)
    sorted_subs = sorted(subtitles, key=lambda s: s.start)
    merged_subs: list[Subtitle] = []
    active: dict[str, Subtitle] = {}

    for res in sorted_subs:
        best_id, best_score = None, 0.0

        for sub_id, sub in active.items():
            # skip if text exceeds time gap tolerance
            if res.start - sub.end > subtitle_config.time_gap_tolerance:
                continue

            # skip if boxes are in different screen regions
            same_box = _is_same_box(
                sub.bbox,
                res.bbox,
                iou_threshold=subtitle_config.box_iou_threshold,
            )
            if not same_box:
                continue

            # skip if text are not the same
            score = SequenceMatcher(None, sub.text, res.text).ratio()
            if score > subtitle_config.text_similarity_threshold and score > best_score:
                best_id = sub_id
                best_score = score

        if best_id:
            sub = active[best_id]
            _update_subtitle(subtitle=sub, result=res)
        else:
            active[str(uuid4())] = res.model_copy()

        to_close = [
            sid
            for sid, s in active.items()
            if res.start - s.end > subtitle_config.time_gap_tolerance
        ]

        for sub_id in to_close:
            merged_subs.append(active.pop(sub_id))

    merged_subs.extend(active.values())
    return _pad_subtitles(
        subtitles=sorted(merged_subs, key=lambda s: s.start),
        meta=meta,
        sub_config=subtitle_config,
    )


def translate(
    video_path: str | Path,
    subtitles: list[Subtitle],
    llm_models: list[LLMModel],
) -> Generator[SSEEvent, None, list[Subtitle]]:
    # Load cache
    video_path = Path(video_path)
    cache = get_settings().cache_dir / video_path.stem / "translated.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        yield LogEvent(message="Using cached translated subtitles")
        return [Subtitle.model_validate(item) for item in data]

    subs_map = {str(idx): sub for idx, sub in enumerate(subtitles)}
    text_map = {str(idx): sub.text for idx, sub in enumerate(subtitles)}
    prompt = json.dumps(text_map, ensure_ascii=False)
    retries = 3

    for attempt in range(1, retries + 1):
        try:
            response = llm_service.complete(
                models=llm_models,
                instruction=TRANSLATE_SUBTITLES,
                prompt=prompt,
            )
            data: dict[str, Any] = json.loads(response)
            translated_subs = [
                sub.model_copy(update={"text": data.get(idx)})
                for idx, sub in subs_map.items()
                if data.get(idx)
            ]

            # Save cache
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(
                json.dumps(
                    [sub.model_dump() for sub in translated_subs],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            yield LogEvent(message="Cached translated subtitles")

            return translated_subs
        except Exception as e:
            logger.warning(
                "Translate subtitles failed - attempt=%s - error=%s",
                attempt,
                e,
            )

            if attempt < retries:
                time.sleep(1)

    raise RuntimeError("Translate subtitles failed after %s attempts", retries)


def _to_srt_timestamp(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(subtitles: list[Subtitle], srt_path: str | Path) -> None:
    """Write a list of subtitles into a file with srt format."""
    srt_path = Path(srt_path)
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    # create subtitle blocks for srt file
    blocks = (
        f"{i}\n{_to_srt_timestamp(s.start)} --> {_to_srt_timestamp(s.end)}\n{s.text}"
        for i, s in enumerate(subtitles, 1)
    )
    # write subtitle blocks to srt file
    srt_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
