from dataclasses import dataclass


@dataclass
class BackgroundConfig:
    residual_weight: float = 0.3
    homography_weight: float = 0.3
    depth_weight: float = 0.4
    # 可选: 启用历史模块（已迁移）区域分析增强前景/背景分离
    enable_region_analysis: bool = False
