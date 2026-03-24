"""可感知主体运动幅度评分器。

基于主体 mask 分离前景/背景光流，计算人类可感知的主体运动强度。
核心思路：观众关注主体运动，而非背景/相机运动。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SubjectMotionDetail:
    """主体运动评分详情。"""

    subject_magnitude: float = 0.0
    """主体区域平均光流幅度 (px/frame)。"""
    background_magnitude: float = 0.0
    """背景区域平均光流幅度 (px/frame)。"""
    perceptual_score: float = 0.0
    """可感知运动得分 0-1（主体运动相对于背景的显著度）。"""
    subject_ratio_mean: float = 0.0
    """平均主体面积占比。"""


def compute_subject_motion_score(
    flows: list[tuple[np.ndarray, np.ndarray]],
    masks: list[np.ndarray],
    subject_ratios: list[float],
) -> tuple[float, SubjectMotionDetail]:
    """从光流 + 主体 mask 计算可感知主体运动幅度。

    Args:
        flows: 光流序列 list[(flow_x, flow_y)]，每帧 shape (H, W)。
        masks: 每帧主体二值 mask (H, W) bool。
        subject_ratios: 每帧主体面积占比。

    Returns:
        (perceptual_score, detail) 元组。
    """
    if not flows or not masks:
        return 0.0, SubjectMotionDetail()

    n = min(len(flows), len(masks))
    subject_mags: list[float] = []
    background_mags: list[float] = []

    for i in range(n):
        flow_x, flow_y = flows[i]
        mag = np.sqrt(flow_x**2 + flow_y**2)
        mask = masks[i].astype(bool)

        # 确保 mask 和 flow 尺寸匹配
        if mask.shape != mag.shape:
            mask = _resize_mask(mask, mag.shape)

        # 主体区域光流
        subject_pixels = mag[mask]
        bg_pixels = mag[~mask]

        if subject_pixels.size > 0:
            subject_mags.append(float(np.mean(subject_pixels)))
        else:
            subject_mags.append(0.0)

        if bg_pixels.size > 0:
            background_mags.append(float(np.mean(bg_pixels)))
        else:
            background_mags.append(0.0)

    mean_subject = float(np.mean(subject_mags)) if subject_mags else 0.0
    mean_background = float(np.mean(background_mags)) if background_mags else 0.0
    mean_ratio = float(np.mean(subject_ratios[:n])) if subject_ratios else 0.0

    # ── 面积归一化: 小主体的运动感知权重更高 ──
    # 直觉: 占画面 5% 的小人跑步，感知上比占 50% 的大脸微动更强
    area_boost = 1.0 / (mean_ratio + 0.1)  # 面积越小 boost 越大，0.1 防除零
    area_boost = float(np.clip(area_boost, 1.0, 5.0))

    boosted_subject = mean_subject * area_boost

    # ── 可感知得分: 主体运动相对于背景的显著度 ──
    eps = 1e-6
    perceptual_raw = boosted_subject / (boosted_subject + mean_background + eps)

    # ── 时序加权: 运动突变帧权重更高 ──
    temporal_weight = _compute_temporal_saliency(subject_mags)
    perceptual_score = float(np.clip(
        perceptual_raw * 0.7 + temporal_weight * 0.3, 0.0, 1.0
    ))

    detail = SubjectMotionDetail(
        subject_magnitude=mean_subject,
        background_magnitude=mean_background,
        perceptual_score=perceptual_score,
        subject_ratio_mean=mean_ratio,
    )
    return perceptual_score, detail


def _compute_temporal_saliency(mags: list[float]) -> float:
    """计算时序显著性: 运动幅度变化越大（加速/减速），感知越强。"""
    if len(mags) < 2:
        return 0.5

    diffs = np.abs(np.diff(mags))
    mean_diff = float(np.mean(diffs))

    # sigmoid 归一化: 突变 > 2 px/frame 视为显著
    return float(1.0 / (1.0 + np.exp(-1.0 * (mean_diff - 2.0))))


def _resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """将 mask resize 到目标 (H, W)。"""
    import cv2

    resized = cv2.resize(
        mask.astype(np.uint8), (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)
