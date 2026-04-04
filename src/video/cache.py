import json
from dataclasses import asdict
from pathlib import Path

from src.core.config import settings
from src.video.types import BoundingBox, Subtitle


def is_exists(input_path: str | Path) -> bool:
    return get_path(input_path).exists()


def get_path(input_path: str | Path) -> Path:
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Cache not found: {input_path}")

    cache_path = settings.cache_dir / f"subtitle{input_path.stem}.json"
    return cache_path


def read(input_path: str | Path) -> list[Subtitle]:
    cache_path = get_path(input_path)

    data = json.loads(cache_path.read_text(encoding="utf-8"))
    subtitles = [
        Subtitle(
            text=item["text"],
            conf=item["conf"],
            bbox=BoundingBox(**item["bbox"]),
            start=item["start"],
            end=item["end"],
        )
        for item in data
    ]
    return subtitles


def write(input_path: str | Path, subtitles: list[Subtitle]):
    cache_path = get_path(input_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache_path.write_text(
        json.dumps([asdict(s) for s in subtitles], ensure_ascii=False),
        encoding="utf-8",
    )
