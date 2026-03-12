from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BiologicalAnomalyConfig:
    """三级生物特征异常检测配置。"""

    # --- Level 1: 快速筛选 ---
    ear_blink_threshold: float = 0.21
    max_no_blink_frames: int = 90
    eye_symmetry_tolerance: float = 0.15
    mar_jump_threshold: float = 0.3
    mar_sustained_open_threshold: float = 0.5
    mar_sustained_open_max_s: float = 3.0
    hand_velocity_threshold: float = 0.3
    hand_jitter_threshold: float = 0.05
    hand_jitter_window: int = 5

    # --- 时序平滑 (借鉴 VMBench OIS 滑窗策略) ---
    smoothing_window: int = 3
    # 相对变化率阈值（替代绝对差分）
    relative_change_threshold: float = 0.45

    # --- Level 2: 结构检测 ---
    finger_count_expected: int = 5
    finger_min_separation: float = 0.02
    finger_fusion_frames: int = 3
    bone_length_ratio_tolerance: float = 0.15
    bone_length_change_tolerance: float = 0.3
    joint_angle_range: tuple[float, float] = (0, 180)
    mouth_area_change_threshold: float = 0.5
    mouth_stability_threshold: float = 0.05

    # --- Level 3: MLLM 兜底 ---
    enable_mllm: bool = True
    mllm_max_crops: int = 8

    # --- 各级权重 ---
    level1_weight: float = 0.3
    level2_weight: float = 0.4
    level3_weight: float = 0.3
