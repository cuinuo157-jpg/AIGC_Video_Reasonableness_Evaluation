from dataclasses import dataclass


@dataclass
class MotionLogicConfig:
    dynamics_weight: float = 0.3
    smoothness_weight: float = 0.4
    naturalness_weight: float = 0.3
    enable_mllm: bool = True
    smoothness_acceleration_weight: float = 0.5
    smoothness_trajectory_weight: float = 0.5
    # Deprecated alias for backward compatibility with old configs.
    smoothness_qalign_weight: float = 0.5
    # Dynamics calibration params (tuned for 512p-ish processing scale).
    flow_threshold_dynamic: float = 5.0
    flow_threshold_static: float = 2.0
    flow_threshold_subject_min: float = 2.0
    flow_subject_relief_factor: float = 0.35
    coverage_motion_threshold: float = 0.5
    temporal_std_threshold: float = 0.5
    camera_score_floor: float = 0.25
