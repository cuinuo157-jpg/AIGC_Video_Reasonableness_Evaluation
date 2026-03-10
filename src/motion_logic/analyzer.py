from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import MotionLogicConfig
from .dynamics_scorer import compute_dynamics_score
from .smoothness_scorer import compute_flow_acceleration_smoothness


@dataclass
class MotionLogicResult:
    applicable: bool
    skip_reason: str | None = None
    dynamics_score: float = 0.0
    smoothness_score: float = 0.0
    naturalness_score: float | None = None
    naturalness_issues: list[str] = field(default_factory=list)
    motion_logic_score: float = 0.0


class MotionLogicAnalyzer:
    def __init__(
        self,
        config: MotionLogicConfig | None = None,
        mllm_client: Any = None,
    ) -> None:
        self.config = config or MotionLogicConfig()
        self._mllm_client = mllm_client

    def analyze(self, hub: Any) -> MotionLogicResult:
        flows = hub.get("optical_flow")
        if not flows or len(flows) < 2:
            return MotionLogicResult(
                applicable=False, skip_reason="no motion detected"
            )

        dynamics = compute_dynamics_score(flows)
        smoothness = compute_flow_acceleration_smoothness(flows)

        naturalness = None
        issues: list[str] = []

        if self.config.enable_mllm and self._mllm_client:
            from .naturalness_judge import judge_naturalness_mllm

            result = judge_naturalness_mllm(
                hub, self._mllm_client, flows, smoothness
            )
            if not result.get("skipped"):
                naturalness = 1.0 if result.get("is_natural", True) else 0.3
                issues = result.get("issues", [])

        c = self.config
        if naturalness is not None:
            score = (
                c.dynamics_weight * dynamics
                + c.smoothness_weight * smoothness
                + c.naturalness_weight * naturalness
            )
        else:
            total_w = c.dynamics_weight + c.smoothness_weight
            score = (
                c.dynamics_weight * dynamics + c.smoothness_weight * smoothness
            ) / total_w

        return MotionLogicResult(
            applicable=True,
            dynamics_score=dynamics,
            smoothness_score=smoothness,
            naturalness_score=naturalness,
            naturalness_issues=issues,
            motion_logic_score=float(np.clip(score, 0, 1)),
        )
