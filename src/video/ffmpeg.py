import json
import subprocess
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np

from src.video.types import VideoMetadata


def _build_probe_cmd(input_path: str | Path) -> list[str]:
    """Build an ffprobe command to extract video stream and format metadata.

    Retrieves dimensions, frame rate, frame count, color properties, pixel
    format, and duration from the first video stream of the given file.

    Args:
        input_path: Path to the input video file.

    Returns:
        Argument list suitable for subprocess execution.
    """

    return [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,"
        "color_range,color_space,color_primaries,color_transfer,pix_fmt"
        ":format=duration",
        "-of",
        "json",
        str(input_path),
    ]


def _build_count_packet_cmd(input_path: str | Path) -> list[str]:
    """Build an ffprobe command to count video packets as a frame-count fallback.

    Used when ``nb_frames`` is unavailable in the stream metadata (e.g. for
    certain container formats that don't store frame counts).

    Args:
        input_path: Path to the input video file.

    Returns:
        Argument list suitable for subprocess execution.
    """

    return [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets",
        "-of",
        "json",
        str(input_path),
    ]


def get_video_metadata(input_path: str | Path) -> VideoMetadata:
    """Probe a video file and return its metadata.

    Runs ffprobe to extract stream properties from the first video track.
    If the container does not report a frame count, a second ffprobe pass
    counts packets instead. Duration falls back to ``total_frames / fps``
    when the format field is missing or zero.

    Args:
        input_path: Path to the input video file.

    Returns:
        A ``VideoMetadata`` instance populated with resolution, frame rate,
        frame count, color properties, pixel format, and duration.

    Raises:
        subprocess.CalledProcessError: If ffprobe exits with a non-zero status.
        KeyError: If expected fields are absent from the ffprobe JSON output.
    """

    result = subprocess.run(
        _build_probe_cmd(input_path),
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    stream: dict[str, Any] = data["streams"][0]

    fps_str: str = stream.get("r_frame_rate", "30/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den)
    total_frames = int(stream.get("nb_frames", 0))
    if total_frames == 0:
        result2 = subprocess.run(
            _build_count_packet_cmd(input_path),
            capture_output=True,
            text=True,
            check=True,
        )
        total_frames = int(json.loads(result2.stdout)["streams"][0]["nb_read_packets"])
    duration = float(data["format"]["duration"]) or total_frames / fps

    return VideoMetadata(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        total_frames=total_frames,
        color_range=stream.get("color_range", "tv"),
        color_space=stream.get("color_space", "bt709"),
        color_primaries=stream.get("color_primaries", "bt709"),
        color_transfer=stream.get("color_transfer", "bt709"),
        pix_fmt=stream.get("pix_fmt", "yuv420p"),
        duration=duration,
    )


def color_flags(metadata: VideoMetadata) -> list[str]:
    """Build ffmpeg color-property flags from video metadata.

    Converts the color fields of a ``VideoMetadata`` object into the
    corresponding ffmpeg CLI flags (``-color_range``, ``-colorspace``,
    ``-color_primaries``, ``-color_trc``). Only flags whose metadata
    field is non-empty are included.

    Args:
        metadata: Video metadata containing color descriptors.

    Returns:
        A flat list of ffmpeg flag/value pairs, e.g.
        ``["-color_range", "tv", "-colorspace", "bt709", ...]``.
    """

    flags = []
    if metadata.color_range:
        flags += ["-color_range", metadata.color_range]
    if metadata.color_space:
        flags += ["-colorspace", metadata.color_space]
    if metadata.color_primaries:
        flags += ["-color_primaries", metadata.color_primaries]
    if metadata.color_transfer:
        flags += ["-color_trc", metadata.color_transfer]
    return flags


def iter_frames(
    input_path: str | Path,
    metadata: VideoMetadata,
) -> Generator[tuple[int, float, np.ndarray], None, None]:
    """Decode a video file and yield raw YUV420p frames one at a time.

    Spawns an ffmpeg subprocess that writes raw ``yuv420p`` frames to stdout,
    then reads and reshapes each frame into a NumPy array. The generator
    performs clean-up (closes the pipe and waits for the process) via a
    ``finally`` block regardless of how iteration ends.

    Args:
        input_path: Path to the input video file.
        metadata: Video metadata used to determine frame size and frame rate.

    Yields:
        A tuple of ``(frame_index, timestamp_seconds, yuv_array)`` where
        ``yuv_array`` has shape ``(height * 3 // 2, width)`` and dtype
        ``uint8``.
    """

    frame_bytes = metadata.width * metadata.height * 3 // 2

    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            *color_flags(metadata),
            "-i",
            str(input_path),
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    frame_idx = 0
    try:
        while True:
            raw = decoder.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break

            yuv = np.frombuffer(raw, dtype=np.uint8).reshape(
                metadata.height * 3 // 2, metadata.width
            )
            yield frame_idx, frame_idx / metadata.fps, yuv
            frame_idx += 1
    finally:
        decoder.stdout.close()
        decoder.wait()


def build_copy_cmd(input_path: str | Path, output_path: str | Path) -> list[str]:
    """Build an ffmpeg command that stream-copies a video without re-encoding.

    All tracks are copied as-is using ``-c copy``, making this a lossless,
    fast remux suitable for container changes or simple file duplication.

    Args:
        input_path: Path to the source video file.
        output_path: Desired path for the output file.

    Returns:
        Argument list suitable for subprocess execution.
    """

    return [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-c",
        "copy",
        str(output_path),
    ]


def _color_bsf_str(metadata: VideoMetadata) -> str:
    """Build an ``hevc_metadata`` bitstream filter string for colour signalling.

    Maps human-readable colour descriptor strings (e.g. ``"bt709"``) to the
    numeric codes required by the HEVC/H.265 ``hevc_metadata`` BSF, and
    assembles them into the ``key=value:key=value`` format expected by ffmpeg's
    ``-bsf:v`` option.

    Args:
        metadata: Video metadata containing colour primaries, transfer
            characteristics, colour space, and colour range.

    Returns:
        A BSF parameter string such as
        ``"hevc_metadata=colour_primaries=1:transfer_characteristics=1:..."``.
    """

    _cp_map = {"bt709": "1", "smpte170m": "6", "bt470bg": "5"}
    _tc_map = {
        "bt709": "1",
        "smpte170m": "6",
        "bt470bg": "5",
        "gamma22": "4",
        "gamma28": "5",
    }
    _mc_map = {"bt709": "1", "smpte170m": "6", "bt470bg": "5"}
    cp = _cp_map.get(metadata.color_primaries, "1")
    tc = _tc_map.get(metadata.color_transfer, "1")
    mc = _mc_map.get(metadata.color_space, "1")
    vfr = "0" if metadata.color_range in ("tv", "limited", "mpeg") else "1"
    return (
        f"hevc_metadata=colour_primaries={cp}"
        f":transfer_characteristics={tc}"
        f":matrix_coefficients={mc}"
        f":video_full_range_flag={vfr}"
    )


def build_encode_cmd(
    input_path: str | Path,
    metadata: VideoMetadata,
    output_path: str | Path,
    video_quality: int = 80,
) -> list[str]:
    """Build an ffmpeg command to encode raw YUV frames to HEVC via VideoToolbox.

    Reads raw ``yuv420p`` video from stdin (``pipe:0``) and audio from
    ``input_path``, encodes video with Apple's ``hevc_videotoolbox`` hardware
    encoder, copies audio unchanged, and writes a fast-start MP4. Colour
    metadata is applied both as ffmpeg stream flags and via the
    ``hevc_metadata`` BSF so that it is preserved in the bitstream itself.

    Args:
        input_path: Path to the original file, used as the audio source.
        metadata: Video metadata for resolution, frame rate, and colour
            properties.
        output_path: Desired path for the encoded output file.
        video_quality: VideoToolbox quality level (0–100); higher means better
            quality. Defaults to ``80``.

    Returns:
        Argument list suitable for subprocess execution, intended to be used
        with ``stdin=subprocess.PIPE`` so that raw frames can be piped in.
    """

    color_range_str = (
        "tv" if metadata.color_range in ("tv", "limited", "mpeg") else "pc"
    )

    # fmt: off
    return [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "yuv420p",
        "-color_range", color_range_str,
        "-colorspace", metadata.color_space or "bt709",
        "-color_primaries", metadata.color_primaries or "bt709",
        "-color_trc", metadata.color_transfer or "bt709",
        "-s", f"{metadata.width}x{metadata.height}",
        "-r", str(metadata.fps),
        "-i", "pipe:0",
        "-i", str(input_path),
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "hevc_videotoolbox",
        "-q:v", str(video_quality),
        "-tag:v", "hvc1",
        "-color_range", color_range_str,
        "-colorspace", metadata.color_space or "bt709",
        "-color_primaries", metadata.color_primaries or "bt709",
        "-color_trc", metadata.color_transfer or "bt709",
        "-bsf:v", _color_bsf_str(metadata=metadata),
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]
    # fmt: on
