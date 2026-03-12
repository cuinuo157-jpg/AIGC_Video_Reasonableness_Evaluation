"""D7: 感知质量维度 — 视频模糊检测与画质评估。

包装 blur_detection/ 中的 Q-Align 模糊检测逻辑，提供与七维度 pipeline 兼容的接口。
Q-Align 模型为可选依赖，不可用时使用 Laplacian 方差降级。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerceptualQualityConfig:
    """感知质量评估配置。"""
    blur_weight: float = 0.5
    consistency_weight: float = 0.3
    artifact_weight: float = 0.2
    # Q-Align 不可用时使用 Laplacian 方差作为降级方案
    use_laplacian_fallback: bool = True
    laplacian_blur_threshold: float = 100.0


@dataclass
class PerceptualQualityResult:
    applicable: bool
    skip_reason: str | None = None
    blur_score: float = 1.0
    consistency_score: float = 1.0
    artifact_score: float = 1.0
    frame_quality_scores: list[float] = field(default_factory=list)
    perceptual_quality_score: float = 1.0


class PerceptualQualityAnalyzer:
    """感知质量分析器 — D7 维度。"""

    def __init__(
        self,
        config: PerceptualQualityConfig | None = None,
    ) -> None:
        self.config = config or PerceptualQualityConfig()

    def analyze(self, hub: Any) -> PerceptualQualityResult:
        """分析视频感知质量。"""
        try:
            frames = hub.get("video_frames")
        except KeyError:
            return PerceptualQualityResult(
                applicable=False, skip_reason="no video frames"
            )

        if not frames or len(frames) < 2:
            return PerceptualQualityResult(
                applicable=False, skip_reason="insufficient frames"
            )

        # 尝试使用 Q-Align 模型
        qalign_available = False
        try:
            from .blur_detection.motion_smoothness_score import QAlignVideoScorer
            qalign_available = True
        except (ImportError, Exception):
            pass

        if qalign_available:
            return self._analyze_with_qalign(frames)
        elif self.config.use_laplacian_fallback:
            return self._analyze_with_laplacian(frames)
        else:
            return PerceptualQualityResult(
                applicable=False, skip_reason="Q-Align not installed"
            )

    def _analyze_with_laplacian(
        self, frames: list[np.ndarray]
    ) -> PerceptualQualityResult:
        """Laplacian 方差降级方案 (无需额外模型)。"""
        quality_scores: list[float] = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            # 归一化: Laplacian 方差越高越清晰
            # 典型范围 0~500+, 使用 sigmoid 归一化
            score = 1.0 / (1.0 + np.exp(-0.02 * (lap_var - self.config.laplacian_blur_threshold)))
            quality_scores.append(float(score))

        if not quality_scores:
            return PerceptualQualityResult(applicable=False, skip_reason="no frames processed")

        # 模糊分: 所有帧的平均质量
        blur_score = float(np.mean(quality_scores))

        # 一致性分: 质量波动越小越好
        if len(quality_scores) > 1:
            consistency_score = float(1.0 - np.clip(np.std(quality_scores) * 2, 0, 1))
        else:
            consistency_score = 1.0

        # 瑕疵分: 质量突降帧占比
        if len(quality_scores) > 2:
            diffs = np.abs(np.diff(quality_scores))
            artifact_ratio = float(np.mean(diffs > 0.3))
            artifact_score = float(1.0 - artifact_ratio)
        else:
            artifact_score = 1.0

        c = self.config
        final = (
            c.blur_weight * blur_score
            + c.consistency_weight * consistency_score
            + c.artifact_weight * artifact_score
        )

        return PerceptualQualityResult(
            applicable=True,
            blur_score=blur_score,
            consistency_score=consistency_score,
            artifact_score=artifact_score,
            frame_quality_scores=quality_scores,
            perceptual_quality_score=float(np.clip(final, 0, 1)),
        )

    def _analyze_with_qalign(
        self, frames: list[np.ndarray]
    ) -> PerceptualQualityResult:
        """Q-Align 模型方案 (高精度)。"""
        try:
            from .blur_detection.motion_smoothness_score import QAlignVideoScorer

            scorer = QAlignVideoScorer()
            # Q-Align 需要 PIL Image 列表
            from PIL import Image

            pil_frames = [
                Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames
            ]
            raw_scores = scorer.score_frames(pil_frames)

            quality_scores = [float(s) for s in raw_scores]
            blur_score = float(np.mean(quality_scores))
            consistency_score = float(
                1.0 - np.clip(np.std(quality_scores) * 2, 0, 1)
            ) if len(quality_scores) > 1 else 1.0

            diffs = np.abs(np.diff(quality_scores)) if len(quality_scores) > 2 else []
            artifact_score = float(1.0 - np.mean(np.array(diffs) > 0.3)) if len(diffs) > 0 else 1.0

            c = self.config
            final = (
                c.blur_weight * blur_score
                + c.consistency_weight * consistency_score
                + c.artifact_weight * artifact_score
            )

            return PerceptualQualityResult(
                applicable=True,
                blur_score=blur_score,
                consistency_score=consistency_score,
                artifact_score=artifact_score,
                frame_quality_scores=quality_scores,
                perceptual_quality_score=float(np.clip(final, 0, 1)),
            )
        except Exception as e:
            logger.warning("Q-Align 分析失败 (%s)，降级到 Laplacian", e)
            return self._analyze_with_laplacian(frames)
