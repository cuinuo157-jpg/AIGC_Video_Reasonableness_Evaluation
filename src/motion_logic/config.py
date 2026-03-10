from dataclasses import dataclass


@dataclass
class MotionLogicConfig:
    dynamics_weight: float = 0.3
    smoothness_weight: float = 0.4
    naturalness_weight: float = 0.3
    enable_mllm: bool = True
    smoothness_acceleration_weight: float = 0.5
    smoothness_qalign_weight: float = 0.5
