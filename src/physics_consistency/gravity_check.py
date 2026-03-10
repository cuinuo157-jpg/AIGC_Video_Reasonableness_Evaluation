from __future__ import annotations

import numpy as np


def check_gravity_consistency(trajectories: list[np.ndarray]) -> list[dict]:
    events = []
    for i, traj in enumerate(trajectories):
        if len(traj) < 5:
            continue
        t = np.arange(len(traj))
        y = traj[:, 1]
        coeffs = np.polyfit(t, y, 2)
        a = coeffs[0]
        residuals = y - np.polyval(coeffs, t)
        fit_error = float(np.mean(residuals**2))
        if a < -0.1 and fit_error < 100:
            events.append(
                {
                    "type": "anti_gravity",
                    "trajectory_idx": i,
                    "acceleration": float(a),
                    "fit_error": fit_error,
                    "description": f"Trajectory {i}: upward acceleration a={a:.3f}",
                }
            )
    return events
