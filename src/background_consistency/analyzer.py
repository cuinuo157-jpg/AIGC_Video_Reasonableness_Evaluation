from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import BackgroundConfig
from .depth_consistency import compute_depth_consistency
from .feature_matching import compute_homography_stability
from .static_region_analysis import compute_residual_score

logger = logging.getLogger(__name__)


@dataclass
class BackgroundConsistencyResult:
    applicable: bool = True
    skip_reason: str | None = None
    residual_score: float = 1.0
    homography_stability: float = 1.0
    depth_consistency: float = 1.0
    region_analysis_score: float | None = None
    background_score: float = 1.0


class BackgroundConsistencyAnalyzer:
    def __init__(self, config: BackgroundConfig | None = None) -> None:
        self.config = config or BackgroundConfig()

    def analyze(self, hub: Any) -> BackgroundConsistencyResult:
        try:
            frames = hub.get("video_frames")
        except KeyError:
            return BackgroundConsistencyResult(
                applicable=False, skip_reason="no video frames"
            )

        residual = compute_residual_score(frames)
        homography = compute_homography_stability(frames)

        depth_score = 1.0
        try:
            depths = hub.get("depth")
            depth_score = compute_depth_consistency(depths)
        except KeyError:
            pass

        # 可选: 区域时序变化分析 (增强前景/背景分离)
        region_score = None
        if self.config.enable_region_analysis:
            region_score = self._try_region_analysis(frames)

        c = self.config
        bg_score = (
            c.residual_weight * residual
            + c.homography_weight * homography
            + c.depth_weight * depth_score
        )

        # 区域分析结果作为加权修正 (如可用)
        if region_score is not None:
            bg_score = 0.85 * bg_score + 0.15 * region_score

        return BackgroundConsistencyResult(
            applicable=True,
            residual_score=residual,
            homography_stability=homography,
            depth_consistency=depth_score,
            region_analysis_score=region_score,
            background_score=float(np.clip(bg_score, 0, 1)),
        )

    def _try_region_analysis(self, frames: list[np.ndarray]) -> float | None:
        """尝试使用 temporal_reasoning 区域分析，不可用时返回 None。"""
        try:
            from src.temporal_reasoning.region_analysis.region_temporal_change_detector import (
                RegionTemporalChangeDetector,
                RegionTemporalChangeConfig,
            )
            from src.temporal_reasoning.motion_flow.flow_analyzer import MotionFlowAnalyzer

            flow_analyzer = MotionFlowAnalyzer()
            detector = RegionTemporalChangeDetector(
                flow_analyzer=flow_analyzer,
                config=RegionTemporalChangeConfig(),
            )
            # 使用全帧作为 "区域"，检测整体时序变化异常
            masks = [None] * len(frames)
            result = detector.analyze(frames, masks, label="background")
            # 从结果提取异常比率，转为分数
            anomalies = result.get("anomaly_count", 0)
            total = max(len(frames) - 1, 1)
            return float(1.0 - min(anomalies / total, 1.0))
        except (ImportError, Exception) as e:
            logger.debug("区域分析不可用: %s", e)
            return None
