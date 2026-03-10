from __future__ import annotations

import numpy as np


def compute_residual_score(
    frames: list[np.ndarray],
    mask: np.ndarray | None = None,
) -> float:
    if len(frames) < 2:
        return 1.0
    ref = frames[0].astype(np.float32)
    residuals = []
    for f in frames[1:]:
        diff = np.abs(f.astype(np.float32) - ref)
        if mask is not None:
            diff = diff[mask]
        residuals.append(float(np.mean(diff)))
    return float(np.clip(1.0 - float(np.mean(residuals)) / 255.0, 0, 1))
