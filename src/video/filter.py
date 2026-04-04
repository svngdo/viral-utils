import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from src.video.config import VideoConfig
from src.video.ffmpeg import build_copy_cmd, build_encode_cmd, iter_frames
from src.video.types import BoundingBox, Subtitle, VideoMetadata


def _inpaint(
    frame: np.ndarray,
    boxes: list[BoundingBox],
    metadata: VideoMetadata,
    config: VideoConfig,
) -> np.ndarray:

    # Convert YUV I420 → BGR so OpenCV can process it
    # ffmpeg outputs raw YUV420p bytes; OpenCV needs BGR to work with
    bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)

    w, h = metadata.width, metadata.height

    for box in boxes:
        # --- Step 1: Expand bounding box ---
        # Grow the region slightly beyond the subtitle box (inpaint_expand pixels)
        # so the inpainter has surrounding context → more natural result
        # Clamp to [0, width/height] to avoid out-of-bounds indexing
        ex1 = max(0, box.x1 - config.inpaint_expand)
        ey1 = max(0, box.y1 - config.inpaint_expand)
        ex2 = min(w, box.x2 + config.inpaint_expand)
        ey2 = min(h, box.y2 + config.inpaint_expand)

        # --- Step 2: Crop ROI (Region of Interest) ---
        # ROI = the subtitle area + surrounding context (expanded above)
        # .copy() prevents modifying the original frame during processing
        roi = bgr[ey1:ey2, ex1:ex2].copy()
        rh, rw = roi.shape[:2]

        # --- Step 3: Scale down for faster inpainting ---
        # cv2.inpaint is slow on large regions; scaling down (e.g. 0.5x) is ~4x faster
        # Tradeoff: slightly lower quality, acceptable for subtitle removal
        sw = max(1, int(rw * config.inpaint_scale))  # max(1,...) prevents zero width
        sh = max(1, int(rh * config.inpaint_scale))
        scaled_roi = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)

        # --- Step 4: Build inpaint mask ---
        # Mask is a grayscale image: 255 = region to inpaint, 0 = keep as-is
        # Only mask the exact subtitle box (b.x1~b.x2), not the expanded region
        # Coordinates must be converted to small_roi space (cropped + scaled)
        mask = np.zeros((sh, sw), dtype=np.uint8)
        mx1 = int((box.x1 - ex1) * config.inpaint_scale)
        my1 = int((box.y1 - ey1) * config.inpaint_scale)
        mx2 = int((box.x2 - ex1) * config.inpaint_scale)
        my2 = int((box.y2 - ey1) * config.inpaint_scale)
        mask[my1:my2, mx1:mx2] = 255

        # --- Step 5: Inpaint ---
        # INPAINT_NS: Navier-Stokes fluid equation, smoother than TELEA for gradient/flat backgrounds
        # inpaintRadius: how many surrounding pixels are used to reconstruct masked area
        inpainted_scaled = cv2.inpaint(
            scaled_roi,
            mask,
            inpaintRadius=config.inpaint_radius,
            flags=cv2.INPAINT_NS,
        )

        # --- Step 6: Scale back to original ROI size ---
        inpainted_full = cv2.resize(
            inpainted_scaled,
            (rw, rh),
            interpolation=cv2.INTER_LINEAR,
        )

        # --- Step 7: Paste result back into frame ---
        # Only paste the exact subtitle box, not the expanded region
        # bx1/by1 are subtitle box coordinates relative to the ROI (not the full frame)
        bx1, by1 = box.x1 - ex1, box.y1 - ey1
        bx2, by2 = box.x2 - ex1, box.y2 - ey1
        bgr[box.y1 : box.y2, box.x1 : box.x2] = inpainted_full[by1:by2, bx1:bx2]

    # --- Step 8: Convert back to YUV to write into ffmpeg pipe ---
    result = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)

    return result


def filter_and_encode(
    input_path: str | Path,
    output_path: str | Path,
    metadata: VideoMetadata,
    subtitles: list[Subtitle],
    config: VideoConfig,
) -> None:
    total_frames = metadata.total_frames

    copy_cmd = build_copy_cmd(input_path, output_path)
    if not subtitles:
        subprocess.run(copy_cmd, check=True)
        return

    encode_cmd = build_encode_cmd(input_path, metadata, output_path)
    encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE)
    try:
        for frame_idx, frame_ts, frame_yuv in iter_frames(input_path, metadata):
            if frame_idx % 10 == 0:
                print(f"Filtered {frame_idx}/{total_frames}", end="\r")

            active_boxes = [
                subtitle.bbox
                for subtitle in subtitles
                if subtitle.start <= frame_ts <= subtitle.end
            ]
            if active_boxes:
                time.sleep(config.inpaint_delay)
                frame_yuv = _inpaint(
                    frame=frame_yuv,
                    boxes=active_boxes,
                    metadata=metadata,
                    config=config,
                )

            encoder.stdin.write(frame_yuv.tobytes())
    finally:
        encoder.stdin.close()
        encoder.wait()
