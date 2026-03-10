from dataclasses import dataclass


@dataclass
class PhysicsConfig:
    drift_flow_threshold: float = 0.5
    drift_min_frames: int = 5
    drift_direction_tolerance: float = 30.0
    gravity_fit_threshold: float = 0.3
    enable_mllm: bool = True
    drift_weight: float = 0.4
    gravity_weight: float = 0.3
    mllm_weight: float = 0.3
