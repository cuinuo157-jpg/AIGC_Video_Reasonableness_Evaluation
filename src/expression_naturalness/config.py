from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExpressionConfig:
    au_smoothness_window: int = 5
    au_jump_threshold: float = 1.5
    flow_consistency_weight: float = 0.3
    au_combination_weight: float = 0.4
    au_smoothness_weight: float = 0.3
