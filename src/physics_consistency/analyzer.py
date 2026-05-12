from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import PhysicsConfig
from .pixel_drift import detect_pixel_drift


@dataclass
class PhysicsConsistencyResult:
    applicable: bool
    skip_reason: str | None = None
    drift_events: list[dict] = field(default_factory=list)
    drift_score: float = 1.0
    vlm_score: float | None = None
    vlm_reasoning: str = ""
    vlm_violations: list[dict] = field(default_factory=list)
    vlm_raw_result: dict[str, Any] | None = None
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

        # 像素漂移检测（辅助信号）
        drift_events = detect_pixel_drift(flows)
        drift_score = (
            1.0
            if not drift_events
            else max(0.0, 1.0 - len(drift_events) * 0.3)
        )

        # VLM 判定（主路径）
        vlm_score = None
        vlm_reasoning = ""
        vlm_violations: list[dict] = []
        vlm_raw_result: dict[str, Any] | None = None

        if self.config.enable_mllm and self._mllm_client:
            from .mllm_physics_judge import judge_physics_mllm

            result = judge_physics_mllm(
                hub, self._mllm_client, drift_events=drift_events or None
            )
            vlm_raw_result = result
            if not result.get("skipped"):
                vlm_score = float(result.get("physics_score", 1.0))
                vlm_score = float(np.clip(vlm_score, 0, 1))
                vlm_reasoning = result.get("reasoning", "")
                vlm_violations = result.get("violations", [])

        # 评分：VLM 可用时取 VLM 评分，否则降级为漂移评分
        if vlm_score is not None:
            physics_score = vlm_score
        else:
            physics_score = self.config.drift_fallback_weight * drift_score

        return PhysicsConsistencyResult(
            applicable=True,
            drift_events=drift_events,
            drift_score=drift_score,
            vlm_score=vlm_score,
            vlm_reasoning=vlm_reasoning,
            vlm_violations=vlm_violations,
            vlm_raw_result=vlm_raw_result,
            physics_score=float(np.clip(physics_score, 0, 1)),
        )
