from __future__ import annotations

import numpy as np


def compute_flow_acceleration_smoothness(
    flows: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    if len(flows) < 3:
        return 1.0
    magnitudes = [float(np.mean(np.sqrt(u**2 + v**2))) for u, v in flows]
    velocities = np.diff(magnitudes)
    accelerations = np.diff(velocities)
    if len(accelerations) == 0:
        return 1.0
    max_acc = np.max(np.abs(accelerations))
    normalized = np.abs(accelerations) / (max_acc + 1e-8)
    return float(np.clip(1.0 - float(np.mean(normalized)), 0, 1))
