from dataclasses import dataclass


@dataclass
class BackgroundConfig:
    residual_weight: float = 0.3
    homography_weight: float = 0.3
    depth_weight: float = 0.4
