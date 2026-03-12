from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import PhysicsConfig
from .gravity_check import check_gravity_consistency
from .pixel_drift import detect_pixel_drift


@dataclass
class PhysicsConsistencyResult:
    applicable: bool
    skip_reason: str | None = None
    drift_events: list[dict] = field(default_factory=list)
    drift_score: float = 1.0
    gravity_violations: list[dict] = field(default_factory=list)
    gravity_score: float = 1.0
    mllm_issues: list[dict] = field(default_factory=list)
    mllm_score: float = 1.0
    physics_score: float = 1.0


class PhysicsConsistencyAnalyzer:
    def __init__(
        self,
        config: PhysicsConfig | None = None,
        mllm_client: Any = None,
    ) -> None:
        self.config = config or PhysicsConfig()
        self._mllm_client = mllm_client

    def analyze(self, hub: Any) -> PhysicsConsistencyResult:
        flows = hub.get("optical_flow")
        if not flows or len(flows) < 2:
            return PhysicsConsistencyResult(
                applicable=False, skip_reason="no motion"
            )

        drift_events = detect_pixel_drift(flows)
        drift_score = (
            1.0
            if not drift_events
            else max(0.0, 1.0 - len(drift_events) * 0.3)
        )

        gravity_violations, gravity_score = [], 1.0
        if self.config.gravity_weight > 0:
            try:
                tracking_data = hub.get("tracking")
                if tracking_data:
                    gravity_violations = check_gravity_consistency(tracking_data)
                    gravity_score = 1.0 if not gravity_violations else 0.3
            except (KeyError, Exception) as e:
                import logging
                logging.getLogger(__name__).warning(
                    "重力检测跳过: tracking extractor 不可用 (%s)", e
                )

        mllm_issues, mllm_score = [], 1.0
        if self.config.enable_mllm and self._mllm_client:
            from .mllm_physics_judge import judge_physics_mllm

            result = judge_physics_mllm(hub, self._mllm_client)
            if not result.get("skipped"):
                mllm_issues = result.get("violations", [])
                mllm_score = 0.3 if result.get("has_violations") else 1.0

        c = self.config
        physics_score = (
            c.drift_weight * drift_score
            + c.gravity_weight * gravity_score
            + c.mllm_weight * mllm_score
        )

        return PhysicsConsistencyResult(
            applicable=True,
            drift_events=drift_events,
            drift_score=drift_score,
            gravity_violations=gravity_violations,
            gravity_score=gravity_score,
            mllm_issues=mllm_issues,
            mllm_score=mllm_score,
            physics_score=float(np.clip(physics_score, 0, 1)),
        )
