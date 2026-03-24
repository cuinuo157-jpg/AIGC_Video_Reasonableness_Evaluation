from dataclasses import dataclass


@dataclass
class TemporalCoherenceConfig:
    """TCS-lite 配置。"""

    sample_interval: int = 5
    min_box_area_ratio: float = 0.001
    iou_match_threshold: float = 0.3
    max_track_gap_steps: int = 1
    edge_margin_ratio: float = 0.08
    min_track_len_steps: int = 2
    shrink_ratio_threshold: float = 0.65
    grow_ratio_threshold: float = 1.35
