from __future__ import annotations

from .anomaly_rules import EYE_CONSTRAINTS


def detect_eye_anomalies(
    ear_sequence: list[float],
    fps: float = 30.0,
    constraints: dict | None = None,
) -> list[dict]:
    c = constraints or EYE_CONSTRAINTS
    threshold = c["ear_blink_threshold"]
    max_no_blink = c["max_no_blink_frames"]
    anomalies: list[dict] = []
    consecutive_open = 0
    for i, ear in enumerate(ear_sequence):
        if ear > threshold:
            consecutive_open += 1
        else:
            consecutive_open = 0
        if consecutive_open >= max_no_blink:
            anomalies.append(
                {
                    "type": "no_blink",
                    "frame_idx": i,
                    "duration_frames": consecutive_open,
                    "description": (
                        f"No blink for {consecutive_open} frames "
                        f"({consecutive_open / fps:.1f}s)"
                    ),
                }
            )
            consecutive_open = 0
    return anomalies
