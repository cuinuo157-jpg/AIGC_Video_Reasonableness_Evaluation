from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import MotionLogicConfig
from .dynamics_scorer import compute_dynamics_score, DynamicsDetail
from .smoothness_scorer import compute_flow_acceleration_smoothness


@dataclass
class MotionLogicResult:
    applicable: bool
    skip_reason: str | None = None
    dynamics_score: float = 0.0
    dynamics_detail: DynamicsDetail | None = None
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

        # 优先使用相机补偿后的残差光流
        camera_magnitude = 0.0
        try:
            cam_result = hub.get("camera_compensation")
            if cam_result and cam_result.residual_flows:
                residual_flows = []
                for rf in cam_result.residual_flows:
                    if rf.ndim == 3 and rf.shape[-1] == 2:
                        residual_flows.append((rf[..., 0], rf[..., 1]))
                    else:
                        residual_flows.append((rf[0], rf[1]) if rf.ndim == 3 else (rf, rf))
                if residual_flows:
                    flows = residual_flows
                camera_magnitude = cam_result.camera_magnitude
        except (KeyError, Exception):
            pass  # camera_compensation 不可用，使用原始光流

        dynamics, detail = compute_dynamics_score(flows, camera_magnitude)
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
            dynamics_detail=detail,
            smoothness_score=smoothness,
            naturalness_score=naturalness,
            naturalness_issues=issues,
            motion_logic_score=float(np.clip(score, 0, 1)),
        )
