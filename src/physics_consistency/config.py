from dataclasses import dataclass


@dataclass
class PhysicsConfig:
    drift_flow_threshold: float = 0.5
    drift_min_frames: int = 5
    drift_direction_tolerance: float = 30.0
    gravity_fit_threshold: float = 0.3
    enable_mllm: bool = True
    # gravity_weight 默认 0: tracking extractor 未注册时重力检测不可用
    # 当 feature_hub 注册了 "tracking" extractor 后可手动恢复为 0.3
    drift_weight: float = 0.55
    gravity_weight: float = 0.0
    mllm_weight: float = 0.45
