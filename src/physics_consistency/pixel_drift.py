from __future__ import annotations

import numpy as np


def detect_pixel_drift(
    flows: list[tuple[np.ndarray, np.ndarray]],
    static_mask: np.ndarray | None = None,
    flow_threshold: float = 0.5,
    min_frames: int = 5,
) -> list[dict]:
    if not flows:
        return []

    events = []

    if static_mask is None:
        avg_mag = np.mean([np.sqrt(u**2 + v**2) for u, v in flows], axis=0)
        static_mask = avg_mag < flow_threshold

    directions, magnitudes = [], []
    for u, v in flows:
        masked_u, masked_v = u[static_mask], v[static_mask]
        if len(masked_u) == 0:
            continue
        mean_u, mean_v = float(np.mean(masked_u)), float(np.mean(masked_v))
        magnitudes.append(np.sqrt(mean_u**2 + mean_v**2))
        directions.append(np.degrees(np.arctan2(mean_v, mean_u)))

    if len(directions) >= min_frames:
        dir_std, avg_mag = float(np.std(directions)), float(np.mean(magnitudes))
        if dir_std < 30.0 and avg_mag > flow_threshold:
            events.append(
                {
                    "type": "pixel_drift",
                    "avg_magnitude": avg_mag,
                    "direction_std": dir_std,
                    "duration_frames": len(directions),
                    "description": f"Persistent drift: avg_mag={avg_mag:.2f}, dir_std={dir_std:.1f}",
                }
            )

    return events
