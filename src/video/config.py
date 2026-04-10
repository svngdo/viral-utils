from dataclasses import dataclass


@dataclass
class VideoConfig:
    # --- OCR ---
    ocr_engine_name: str = "ocrmac"
    ocr_sample_interval: int = 3
    ocr_chinese_only: bool = True
    ocr_delay: float = 0.36  # seconds

    # --- Subtitle ---
    sub_frame_gap_tolerance: int = 30
    sub_text_similarity_threshold: float = 0.6
    sub_box_iou_threshold: float = 0.6
    sub_frame_padding: int = 3

    # --- Translate ---
    translate_conf_threshold: float = 0.5

    # --- Inpaint ---
    inpaint_conf_threshold: float = 0.3
    inpaint_scale: float = 0.6
    inpaint_expand: int = 6
    inpaint_radius: int = 6
    inpaint_delay: float = 0.06
