import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from src.logging import get_logger
from src.video.config import VideoConfig
from src.video.ffmpeg import build_copy_cmd, build_encode_cmd, iter_frames
from src.video.schemas import BoundingBox, Subtitle, VideoMetadata

logger = get_logger(__name__)


def _inpaint_frame(
    frame: np.ndarray,
    boxes: list[BoundingBox],
    metadata: VideoMetadata,
    config: VideoConfig,
) -> np.ndarray:
    # YUV I420 -> BGR
    bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
    w, h = metadata.width, metadata.height

    for b in boxes:
        ex1 = max(0, b.x1 - config.inpaint_expand)
        ey1 = max(0, b.y1 - config.inpaint_expand)
        ex2 = min(w, b.x2 + config.inpaint_expand)
        ey2 = min(h, b.y2 + config.inpaint_expand)

        cx1 = max(b.x1, ex1)
        cy1 = max(b.y1, ey1)
        cx2 = min(b.x2, ex2)
        cy2 = min(b.y2, ey2)

        if cx2 <= cx1 or cy2 <= cy1:
            continue

        roi = bgr[ey1:ey2, ex1:ex2].copy()
        rh, rw = roi.shape[:2]

        sw = max(1, int(rw * config.inpaint_scale))
        sh = max(1, int(rh * config.inpaint_scale))
        scaled_roi = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)

        mask = np.zeros((sh, sw), dtype=np.uint8)
        mx1 = int((cx1 - ex1) * config.inpaint_scale)
        my1 = int((cy1 - ey1) * config.inpaint_scale)
        mx2 = int((cx2 - ex1) * config.inpaint_scale)
        my2 = int((cy2 - ey1) * config.inpaint_scale)
        mask[my1:my2, mx1:mx2] = 255

        inpainted = cv2.resize(
            cv2.inpaint(
                scaled_roi,
                mask,
                inpaintRadius=config.inpaint_radius,
                flags=cv2.INPAINT_TELEA,
            ),
            (rw, rh),
            interpolation=cv2.INTER_CUBIC,
        )

        bx1 = cx1 - ex1
        by1 = cy1 - ey1
        bx2 = cx2 - ex1
        by2 = cy2 - ey1

        bgr[cy1:cy2, cx1:cx2] = inpainted[by1:by2, bx1:bx2]

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)


def inpaint_and_encode(
    video_path: str | Path,
    output_path: str | Path,
    metadata: VideoMetadata,
    subtitles: list[Subtitle],
    config: VideoConfig,
) -> None:
    subtitles = [s for s in subtitles if s.conf >= config.inpaint_conf_threshold]
    total_frames = metadata.total_frames
    copy_cmd = build_copy_cmd(video_path, output_path)

    if not subtitles:
        subprocess.run(copy_cmd, check=True)
        return

    encode_cmd = build_encode_cmd(video_path, metadata, output_path)
    encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE)

    try:
        for frame_idx, frame_ts, frame_yuv in iter_frames(video_path, metadata):
            active_boxes = [
                subtitle.bbox
                for subtitle in subtitles
                if subtitle.start <= frame_ts <= subtitle.end
            ]

            if active_boxes:
                frame_yuv = _inpaint_frame(
                    frame=frame_yuv,
                    boxes=active_boxes,
                    metadata=metadata,
                    config=config,
                )

                time.sleep(config.inpaint_delay)
            encoder.stdin.write(frame_yuv.tobytes())

            print(f"Inpainted {frame_idx}/{total_frames}", end="\r")
    finally:
        encoder.stdin.close()
        encoder.wait()

    logger.info(f"Video encoded to: {output_path}")
