from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import BackgroundConfig
from .depth_consistency import compute_depth_consistency
from .feature_matching import compute_homography_stability
from .static_region_analysis import compute_residual_score


@dataclass
class BackgroundConsistencyResult:
    applicable: bool = True
    skip_reason: str | None = None
    residual_score: float = 1.0
    homography_stability: float = 1.0
    depth_consistency: float = 1.0
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

        c = self.config
        bg_score = (
            c.residual_weight * residual
            + c.homography_weight * homography
            + c.depth_weight * depth_score
        )

        return BackgroundConsistencyResult(
            applicable=True,
            residual_score=residual,
            homography_stability=homography,
            depth_consistency=depth_score,
            background_score=float(np.clip(bg_score, 0, 1)),
        )
