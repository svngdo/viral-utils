import threading

import pytest

from src.inpainting import service as inpainting_service
from src.job.exceptions import JobCancelled
from src.job.schemas import (
    JobStatus,
    LogEvent,
    ProgressEvent,
    StatusEvent,
)
from src.ocr import service as ocr_service
from src.video import service as video_service
from src.video.schemas import ProcessVideosRequest
from src.video.service import process


class FailsIfTouchedVideoEngine:
    def get_metadata(self, *_args, **_kwargs):
        raise AssertionError("video metadata should not be read after cancellation")

    def copy(self, *_args, **_kwargs):
        raise AssertionError("video should not be copied after cancellation")

    def get_encoder(self, *_args, **_kwargs):
        raise AssertionError("encoder should not start after cancellation")


class CopyRecordingVideoEngine:
    def __init__(self):
        self.copied = []

    def copy(self, src, dst):
        self.copied.append((src, dst))


def test_process_returns_completed_when_input_dir_has_no_videos(tmp_path):
    request = ProcessVideosRequest(
        in_dir=str(tmp_path),
        out_dir=str(tmp_path / "out"),
    )

    events = list(
        process(
            request=request,
            video_engine=object(),
            ocr_engine=object(),
            ocr_config=object(),
            subtitle_config=object(),
            inpaint_engine=object(),
            inpaint_config=object(),
            llm_models=request.llm_models,
        )
    )

    assert len(events) == 2
    assert events[0] == LogEvent(message="No videos to process", id=events[0].id)
    assert events[1].status == JobStatus.COMPLETED


def test_ocr_cancelled_before_metadata_read(tmp_path):
    cancel = threading.Event()
    cancel.set()

    events = ocr_service.extract(
        video_path=tmp_path / "video.mp4",
        video_engine=FailsIfTouchedVideoEngine(),
        engine=object(),
        config=object(),
        cancel=cancel,
    )

    with pytest.raises(JobCancelled):
        next(events)


def test_inpaint_cancelled_before_ffmpeg_work(tmp_path):
    cancel = threading.Event()
    cancel.set()

    events = inpainting_service.inpaint(
        video_path=tmp_path / "video.mp4",
        out_path=tmp_path / "out.mp4",
        subtitles=[],
        video_engine=FailsIfTouchedVideoEngine(),
        engine=object(),
        config=object(),
        cancel=cancel,
    )

    with pytest.raises(JobCancelled):
        next(events)


def test_inpaint_logs_when_output_file_already_exists(tmp_path):
    out_path = tmp_path / "out.mp4"
    out_path.write_text("processed", encoding="utf-8")

    events = list(
        inpainting_service.inpaint(
            video_path=tmp_path / "video.mp4",
            out_path=out_path,
            subtitles=[object()],
            video_engine=FailsIfTouchedVideoEngine(),
            engine=object(),
            config=object(),
        )
    )

    assert events == [
        LogEvent(message=f"File already processed: {out_path}", id=events[0].id)
    ]


def test_process_emits_video_level_progress_for_web_ui(tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_text("video", encoding="utf-8")
    request = ProcessVideosRequest(
        in_dir=str(tmp_path),
        out_dir=str(tmp_path / "out"),
    )

    def extract(**_kwargs):
        yield LogEvent(message="OCR: 10/100 frames")
        return [object()]

    def inpaint(**_kwargs):
        yield LogEvent(message="Inpainting: 20/100 frames")
        yield LogEvent(message="Saved processed video: out/video.mp4")
        yield LogEvent(message="Inpainting completed for video.mp4")

    def translate(**_kwargs):
        yield LogEvent(message="Saved translated subtitles cache for video.mp4")
        return []

    monkeypatch.setattr(video_service.ocr_service, "extract", extract)
    monkeypatch.setattr(video_service.subtitle_service, "merge", lambda **_kwargs: [])
    monkeypatch.setattr(video_service.subtitle_service, "translate", translate)
    monkeypatch.setattr(video_service.subtitle_service, "write_srt", lambda **_kwargs: None)
    monkeypatch.setattr(video_service.inpaint_service, "inpaint", inpaint)

    events = list(
        process(
            request=request,
            video_engine=object(),
            ocr_engine=object(),
            ocr_config=object(),
            subtitle_config=object(),
            inpaint_engine=object(),
            inpaint_config=object(),
            llm_models=request.llm_models,
        )
    )

    progress_events = [event for event in events if event.type == "progress"]
    assert [(event.done, event.total) for event in progress_events] == [(0, 1), (1, 1)]

    completed_events = [
        event
        for event in events
        if isinstance(event, StatusEvent) and event.status == JobStatus.COMPLETED
    ]
    assert len(completed_events) == 1

    assert any(
        event.type == "log" and event.message == "OCR: 10/100 frames"
        for event in events
    )
    assert any(
        event.type == "log" and event.message == "Inpainting: 20/100 frames"
        for event in events
    )


def test_process_saves_last_video_when_no_subtitles_are_found(tmp_path, monkeypatch):
    first_video = tmp_path / "01-video.mp4"
    last_video = tmp_path / "02-video.mp4"
    first_video.write_text("video", encoding="utf-8")
    last_video.write_text("video", encoding="utf-8")
    out_dir = tmp_path / "out"
    request = ProcessVideosRequest(
        in_dir=str(tmp_path),
        out_dir=str(out_dir),
    )
    video_engine = CopyRecordingVideoEngine()

    def extract(video_path, **_kwargs):
        if video_path == last_video:
            return []
        yield ProgressEvent(done=10, total=100)
        return [object()]

    def inpaint(**_kwargs):
        yield LogEvent(message=f"Saved processed video: {out_dir / first_video.name}")
        yield LogEvent(message=f"Inpainting completed for {first_video.name}")

    def translate(**_kwargs):
        yield LogEvent(message=f"Saved translated subtitles cache for {first_video.name}")
        return []

    monkeypatch.setattr(video_service.ocr_service, "extract", extract)
    monkeypatch.setattr(video_service.subtitle_service, "merge", lambda **_kwargs: [])
    monkeypatch.setattr(video_service.subtitle_service, "translate", translate)
    monkeypatch.setattr(video_service.subtitle_service, "write_srt", lambda **_kwargs: None)
    monkeypatch.setattr(video_service.inpaint_service, "inpaint", inpaint)

    events = list(
        process(
            request=request,
            video_engine=video_engine,
            ocr_engine=object(),
            ocr_config=object(),
            subtitle_config=object(),
            inpaint_engine=object(),
            inpaint_config=object(),
            llm_models=request.llm_models,
        )
    )

    assert video_engine.copied == [(last_video, out_dir / last_video.name)]
    assert any(
        event.type == "log"
        and event.message == f"Saved processed video: {out_dir / last_video.name}"
        for event in events
    )
    assert [event.done for event in events if event.type == "progress"][-1] == 2
    assert events[-1].status == JobStatus.COMPLETED


def test_process_raises_job_cancelled_at_video_boundary(tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.write_text("video", encoding="utf-8")
    request = ProcessVideosRequest(
        in_dir=str(tmp_path),
        out_dir=str(tmp_path / "out"),
    )
    cancel = threading.Event()
    cancel.set()

    events = process(
        request=request,
        video_engine=object(),
        ocr_engine=object(),
        ocr_config=object(),
        subtitle_config=object(),
        inpaint_engine=object(),
        inpaint_config=object(),
        llm_models=request.llm_models,
        cancel=cancel,
    )

    assert next(events).status == JobStatus.RUNNING
    assert next(events).message == "Processing 1 videos"
    assert next(events).done == 0
    with pytest.raises(JobCancelled):
        next(events)
