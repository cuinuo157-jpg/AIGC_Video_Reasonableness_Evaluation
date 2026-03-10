from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FaceIdentityConfig:
    csim_ref_weight: float = 0.4
    csim_adj_weight: float = 0.3
    csim_min_weight: float = 0.2
    drop_penalty_weight: float = 0.1
    drop_threshold: float = 0.3
    drop_window: int = 3
