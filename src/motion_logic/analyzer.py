from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import MotionLogicConfig
from .dynamics_scorer import compute_dynamics_score, DynamicsDetail
from .smoothness_scorer import compute_flow_acceleration_smoothness
from .subject_motion_scorer import compute_subject_motion_score, SubjectMotionDetail
from .trajectory_curvature_scorer import (
    TrajectoryCurvatureDetail,
    compute_trajectory_curvature_smoothness,
)


@dataclass
class MotionLogicResult:
    applicable: bool
    skip_reason: str | None = None
    dynamics_score: float = 0.0
    dynamics_detail: DynamicsDetail | None = None
    smoothness_score: float = 0.0
    naturalness_score: float | None = None
    naturalness_issues: list[str] = field(default_factory=list)
    naturalness_mllm_result: dict[str, Any] | None = None
    subject_motion_detail: SubjectMotionDetail | None = None
    flow_smoothness_score: float = 0.0
    trajectory_curvature_score: float | None = None
    trajectory_curvature_detail: TrajectoryCurvatureDetail | None = None
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
        # 优先使用 RAFT 光流 (亚像素精度)，不可用时降级为 Farneback
        if hub.has_extractor("raft_flow"):
            flows = hub.get("raft_flow")
        else:
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

        # 尝试获取主体 mask 并计算可感知运动分数
        subject_detail: SubjectMotionDetail | None = None
        try:
            if hub.has_extractor("subject_masks"):
                seg_result = hub.get("subject_masks")
                if seg_result and seg_result.masks and seg_result.method != "none":
                    _, subject_detail = compute_subject_motion_score(
                        flows, seg_result.masks, seg_result.subject_ratios,
                    )
        except (KeyError, Exception):
            pass  # subject_masks 不可用，退回全局模式

        dynamics, detail = compute_dynamics_score(
            flows,
            camera_magnitude,
            subject_motion=subject_detail,
            flow_threshold_dynamic=getattr(self.config, "flow_threshold_dynamic", 5.0),
            flow_threshold_static=getattr(self.config, "flow_threshold_static", 2.0),
            flow_threshold_subject_min=getattr(self.config, "flow_threshold_subject_min", 2.0),
            flow_subject_relief_factor=getattr(self.config, "flow_subject_relief_factor", 0.35),
            coverage_motion_threshold=getattr(self.config, "coverage_motion_threshold", 0.5),
            temporal_std_threshold=getattr(self.config, "temporal_std_threshold", 0.5),
            camera_score_floor=getattr(self.config, "camera_score_floor", 0.25),
        )
        flow_smoothness = compute_flow_acceleration_smoothness(flows)

        trajectory_score: float | None = None
        trajectory_detail: TrajectoryCurvatureDetail | None = None
        try:
            if hub.has_extractor("tracking"):
                trajectories = hub.get("tracking")
                if trajectories:
                    trajectory_score, trajectory_detail = compute_trajectory_curvature_smoothness(
                        trajectories
                    )
        except (KeyError, Exception):
            pass

        if trajectory_score is not None:
            acc_w = self.config.smoothness_acceleration_weight
            traj_w = getattr(
                self.config,
                "smoothness_trajectory_weight",
                getattr(self.config, "smoothness_qalign_weight", 0.5),
            )
            total = max(acc_w + traj_w, 1e-8)
            smoothness = (acc_w * flow_smoothness + traj_w * trajectory_score) / total
        else:
            smoothness = flow_smoothness

        naturalness = None
        issues: list[str] = []
        naturalness_mllm_result: dict[str, Any] | None = None

        if self.config.enable_mllm and self._mllm_client:
            from .naturalness_judge import judge_naturalness_mllm

            result = judge_naturalness_mllm(
                hub,
                self._mllm_client,
                flows,
                smoothness,
                smoothness_threshold=getattr(
                    self.config, "naturalness_smoothness_threshold", 0.8
                ),
            )
            naturalness_mllm_result = result
            if not result.get("skipped"):
                is_natural = result.get("is_natural")
                if is_natural is None:
                    is_natural = result.get("is_reasonable")
                if is_natural is None:
                    is_natural = True
                naturalness = 1.0 if is_natural else 0.3
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
            naturalness_mllm_result=naturalness_mllm_result,
            subject_motion_detail=subject_detail,
            flow_smoothness_score=flow_smoothness,
            trajectory_curvature_score=trajectory_score,
            trajectory_curvature_detail=trajectory_detail,
            motion_logic_score=float(np.clip(score, 0, 1)),
        )
