from __future__ import annotations

import numpy as np


def compute_dynamics_score(
    flows: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    if not flows:
        return 0.0
    magnitudes = [float(np.mean(np.sqrt(u**2 + v**2))) for u, v in flows]
    return float(np.clip(np.mean(magnitudes) / 10.0, 0, 1))
