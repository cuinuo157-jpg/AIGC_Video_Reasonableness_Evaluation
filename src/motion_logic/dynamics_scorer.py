"""动态度评分器 — 5+1 分量加权融合。

基础 5 分量算法：
  flow_magnitude, spatial_coverage, temporal_variation,
  spatial_consistency, camera_factor

可选第 6 分量（需主体 mask）：
  subject_perceptual — 可感知主体运动得分
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .subject_motion_scorer import SubjectMotionDetail


@dataclass
class DynamicsDetail:
    """动态度评分详情。"""
    unified_score: float = 0.0
    flow_magnitude: float = 0.0
    spatial_coverage: float = 0.0
    temporal_variation: float = 0.0
    spatial_consistency: float = 0.0
    camera_factor: float = 0.5
    subject_perceptual: float | None = None
    scene_type: str = "dynamic"
    interpretation: str = ""


def _sigmoid(value: float, threshold: float, steepness: float = 0.5) -> float:
    return 1.0 / (1.0 + np.exp(-steepness * (value - threshold)))


def compute_dynamics_score(
    flows: list[tuple[np.ndarray, np.ndarray]],
    camera_magnitude: float = 0.0,
    subject_motion: SubjectMotionDetail | None = None,
) -> tuple[float, DynamicsDetail]:
    """从光流序列计算动态度评分。

    Args:
        flows: 光流序列 list[(flow_x, flow_y)]，每帧 shape (H, W)。
        camera_magnitude: 相机运动幅度 (来自 camera_compensation extractor)。
        subject_motion: 主体运动详情 (来自 subject_motion_scorer)，可选。

    Returns:
        (unified_score, detail) 元组。
    """
    if not flows:
        return 0.0, DynamicsDetail()

    # --- 1. 逐帧统计 ---
    frame_magnitudes: list[float] = []
    frame_dynamic_ratios: list[float] = []
    frame_consistency: list[float] = []

    for flow_x, flow_y in flows:
        mag = np.sqrt(flow_x**2 + flow_y**2)
        mean_mag = float(np.mean(mag))
        frame_magnitudes.append(mean_mag)

        # 空间覆盖率: 运动像素占比 (mag > 1.0 pixel)
        dynamic_ratio = float(np.mean(mag > 1.0))
        frame_dynamic_ratios.append(dynamic_ratio)

        # 空间一致性: 运动方向标准差 (越小越一致)
        angle = np.arctan2(flow_y, flow_x + 1e-8)
        moving_mask = mag > 0.5
        if np.any(moving_mask):
            angle_std = float(np.std(angle[moving_mask]))
        else:
            angle_std = 0.0
        frame_consistency.append(angle_std)

    mean_mag = float(np.mean(frame_magnitudes))
    std_mag = float(np.std(frame_magnitudes))
    mean_coverage = float(np.mean(frame_dynamic_ratios))
    mean_consistency = float(np.mean(frame_consistency))

    # --- 2. 场景类型检测 ---
    static_ratio = 1.0 - mean_coverage
    scene_type = "static" if (camera_magnitude > 0.5 and static_ratio > 0.5) else "dynamic"

    # --- 3. 分量评分 ---
    # 3a. 光流幅度 (有主体信息时用主体运动修正阈值)
    if subject_motion is not None and subject_motion.subject_magnitude > 0:
        # 主体运动幅度作为参考，降低全局阈值要求
        adjusted_threshold = max(5.0, 15.0 - subject_motion.subject_magnitude * 0.5)
        flow_score = float(np.clip(
            _sigmoid(mean_mag, threshold=adjusted_threshold, steepness=0.3), 0, 1
        ))
    elif scene_type == "static":
        flow_score = float(np.clip(_sigmoid(mean_mag, threshold=5.0, steepness=0.5), 0, 1))
    else:
        flow_score = float(np.clip(_sigmoid(mean_mag, threshold=15.0, steepness=0.3), 0, 1))

    # 3b. 空间覆盖
    spatial_score = float(np.clip(mean_coverage, 0, 1))

    # 3c. 时序变化
    temporal_score = float(np.clip(_sigmoid(std_mag, threshold=1.0, steepness=1.0), 0, 1))

    # 3d. 空间一致性 (方向一致 → 高分 → 可能是相机运动)
    consistency_score = float(np.clip(1.0 - mean_consistency / np.pi, 0, 1))

    # 3e. 相机因子
    if camera_magnitude > 0:
        camera_score = float(np.clip(1.0 - camera_magnitude / (mean_mag + 1e-6), 0, 1))
    else:
        camera_score = 0.5

    # 3f. 主体可感知运动 (可选)
    subject_score = None
    if subject_motion is not None:
        subject_score = subject_motion.perceptual_score

    # --- 4. 场景自适应加权融合 ---
    has_subject = subject_score is not None

    if has_subject:
        # 6 分量模式: subject_perceptual 占 20%，其余按比例缩减
        if scene_type == "static":
            weights = {
                "flow_magnitude": 0.30,
                "spatial_coverage": 0.20,
                "temporal_variation": 0.10,
                "spatial_consistency": 0.05,
                "camera_factor": 0.15,
                "subject_perceptual": 0.20,
            }
        else:
            weights = {
                "flow_magnitude": 0.30,
                "spatial_coverage": 0.20,
                "temporal_variation": 0.15,
                "spatial_consistency": 0.05,
                "camera_factor": 0.10,
                "subject_perceptual": 0.20,
            }
        scores = {
            "flow_magnitude": flow_score,
            "spatial_coverage": spatial_score,
            "temporal_variation": temporal_score,
            "spatial_consistency": consistency_score,
            "camera_factor": camera_score,
            "subject_perceptual": subject_score,
        }
    else:
        # 原始 5 分量模式
        if scene_type == "static":
            weights = {
                "flow_magnitude": 0.45,
                "spatial_coverage": 0.30,
                "temporal_variation": 0.10,
                "spatial_consistency": 0.05,
                "camera_factor": 0.10,
            }
        else:
            weights = {
                "flow_magnitude": 0.45,
                "spatial_coverage": 0.30,
                "temporal_variation": 0.15,
                "spatial_consistency": 0.05,
                "camera_factor": 0.05,
            }
        scores = {
            "flow_magnitude": flow_score,
            "spatial_coverage": spatial_score,
            "temporal_variation": temporal_score,
            "spatial_consistency": consistency_score,
            "camera_factor": camera_score,
        }

    unified = sum(scores[k] * weights[k] for k in scores)
    unified = float(np.clip(unified, 0, 1))

    # --- 5. 语义解释 ---
    if unified < 0.2:
        level = "极低动态（纯静态）"
    elif unified < 0.4:
        level = "低动态"
    elif unified < 0.6:
        level = "中等动态"
    elif unified < 0.8:
        level = "高动态"
    else:
        level = "极高动态"

    detail = DynamicsDetail(
        unified_score=unified,
        flow_magnitude=flow_score,
        spatial_coverage=spatial_score,
        temporal_variation=temporal_score,
        spatial_consistency=consistency_score,
        camera_factor=camera_score,
        subject_perceptual=subject_score,
        scene_type=scene_type,
        interpretation=f"动态度: {unified:.3f} ({level}), 场景: {scene_type}",
    )

    return unified, detail
