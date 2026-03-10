from __future__ import annotations

from .anomaly_rules import HAND_CONSTRAINTS


def detect_hand_anomalies(
    finger_counts: list[int] | None = None,
    joint_angles: list[list[float]] | None = None,
) -> list[dict]:
    anomalies: list[dict] = []
    expected = HAND_CONSTRAINTS["finger_count"]
    if finger_counts:
        for i, count in enumerate(finger_counts):
            if count != expected:
                anomalies.append(
                    {
                        "type": "wrong_finger_count",
                        "frame_idx": i,
                        "expected": expected,
                        "actual": count,
                        "description": (
                            f"Frame {i}: {count} fingers (expected {expected})"
                        ),
                    }
                )
    if joint_angles:
        lo, hi = HAND_CONSTRAINTS["joint_angle_range"]
        for i, angles in enumerate(joint_angles):
            for j, angle in enumerate(angles):
                if angle < lo or angle > hi:
                    anomalies.append(
                        {
                            "type": "impossible_joint_angle",
                            "frame_idx": i,
                            "joint_idx": j,
                            "angle": angle,
                            "description": (
                                f"Frame {i}: joint {j} angle {angle:.1f} "
                                f"out of range [{lo}, {hi}]"
                            ),
                        }
                    )
    return anomalies
