from __future__ import annotations

import numpy as np


def compute_depth_consistency(depth_maps: list[np.ndarray]) -> float:
    if len(depth_maps) < 2:
        return 1.0
    correlations = []
    for i in range(len(depth_maps) - 1):
        d1, d2 = depth_maps[i].flatten(), depth_maps[i + 1].flatten()
        if np.std(d1) < 1e-8 or np.std(d2) < 1e-8:
            correlations.append(1.0 if np.allclose(d1, d2) else 0.0)
            continue
        correlations.append(max(float(np.corrcoef(d1, d2)[0, 1]), 0.0))
    return float(np.mean(correlations))
