import json
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.llm.client import LLMClient
from src.video.config import VideoConfig
from src.video.constants import TRANSLATE_SUBTITLE_SYSTEM_PROMPT
from src.video.types import BoundingBox, Subtitle, VideoMetadata


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
    subtitles: list[Subtitle],
    metadata: VideoMetadata,
    config: VideoConfig,
):
    frame_duration = 1 / metadata.fps
    padding_ts = frame_duration * config.sub_frame_padding
    return [
        replace(
            s,
            start=max(s.start - padding_ts, 0),
            end=min(s.end + padding_ts, metadata.duration),
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


def merge_subtitles(
    ocr_results: list[Subtitle],
    metadata: VideoMetadata,
    config: VideoConfig,
) -> list[Subtitle]:
    if not ocr_results:
        return []

    fps = metadata.fps
    time_gap_tolerance = config.sub_frame_gap_tolerance / fps
    active: dict[str, Subtitle] = {}
    closed: list[Subtitle] = []

    for res in ocr_results:
        best_id, best_score = None, 0.0

        for sub_id, sub in active.items():
            # skip if text exceeds time gap tolerance
            if res.start - sub.end > time_gap_tolerance:
                continue

            # skip if boxes are in different screen regions
            if not _is_same_box(
                sub.bbox, res.bbox, iou_threshold=config.sub_box_iou_threshold
            ):
                continue

            # skip if text are not the same
            score = SequenceMatcher(None, sub.text, res.text).ratio()
            if score > config.sub_text_similarity_threshold and score > best_score:
                best_id = sub_id
                best_score = score

        if best_id:
            sub = active[best_id]
            _update_subtitle(subtitle=sub, result=res)
        else:
            active[str(uuid4())] = replace(res)

        to_close = [
            sid for sid, s in active.items() if res.start - s.end > time_gap_tolerance
        ]

        for sub_id in to_close:
            closed.append(active.pop(sub_id))

    closed.extend(active.values())

    sorted_subs = sorted(closed, key=lambda s: s.start)

    padded = _pad_subtitles(
        subtitles=sorted_subs,
        metadata=metadata,
        config=config,
    )
    return padded


def _to_srt_timestamp(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def translate_subtitle(
    subtitles: list[Subtitle],
    llm_client: LLMClient,
) -> list[Subtitle]:
    subtitle_map = {str(i): s for i, s in enumerate(subtitles)}
    text_map = {str(i): s.text.strip() for i, s in enumerate(subtitles)}
    content = json.dumps(text_map, ensure_ascii=False)

    response = llm_client.complete(
        system_prompt=TRANSLATE_SUBTITLE_SYSTEM_PROMPT,
        content=content,
    )

    data: dict[str, Any] = json.loads(response)

    return [
        replace(s, text=data.get(idx), bbox=replace(s.bbox))
        for idx, s in subtitle_map.items()
        if data.get(idx)
    ]


def write_srt(subtitles: list[Subtitle], srt_path: str | Path) -> None:
    srt_path = Path(srt_path)
    srt_path.parent.mkdir(parents=True, exist_ok=True)

    blocks = (
        f"{i}\n{_to_srt_timestamp(s.start)} --> {_to_srt_timestamp(s.end)}\n{s.text}"
        for i, s in enumerate(subtitles, 1)
    )

    srt_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
