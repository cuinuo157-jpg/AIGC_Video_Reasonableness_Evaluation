from dataclasses import dataclass


@dataclass
class PhysicsConfig:
    # 像素漂移（辅助信号，不直接参与评分，仅作为 VLM 上下文）
    drift_flow_threshold: float = 0.5
    drift_min_frames: int = 5
    drift_direction_tolerance: float = 30.0
    # VLM 判定
    enable_mllm: bool = True
    # 无 VLM 时的降级评分权重
    drift_fallback_weight: float = 1.0
