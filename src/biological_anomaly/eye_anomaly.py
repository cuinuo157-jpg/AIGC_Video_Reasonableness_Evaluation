from __future__ import annotations

from .anomaly_rules import EYE_CONSTRAINTS, sliding_window_smooth


def detect_eye_anomalies(
    ear_sequence: list[float | None],
    fps: float = 30.0,
    constraints: dict | None = None,
    smoothing_window: int = 3,
) -> list[dict]:
    """Level 1: 基于 EAR 序列检测眼部异常。

    检测项：
    - 长时间未眨眼（no_blink）
    """
    c = constraints or EYE_CONSTRAINTS
    threshold = c["ear_blink_threshold"]
    max_no_blink = c["max_no_blink_frames"]
    anomalies: list[dict] = []
    consecutive_open = 0

    # 滑窗平滑降低噪声误报
    ear_sequence = sliding_window_smooth(ear_sequence, smoothing_window)

    for i, ear in enumerate(ear_sequence):
        if ear is None:
            consecutive_open = 0
            continue
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
                    "severity": "medium",
                    "confidence": 0.8,
                    "description": (
                        f"No blink for {consecutive_open} frames "
                        f"({consecutive_open / fps:.1f}s)"
                    ),
                }
            )
            consecutive_open = 0

    return anomalies


def detect_eye_symmetry_anomalies(
    ear_left_seq: list[float | None],
    ear_right_seq: list[float | None],
    fps: float = 30.0,
    tolerance: float = 0.15,
    smoothing_window: int = 3,
) -> list[dict]:
    """Level 1: 检测左右眼 EAR 不对称（单眼异常闭合/张开）。"""
    # 滑窗平滑降低噪声误报
    ear_left_seq = sliding_window_smooth(ear_left_seq, smoothing_window)
    ear_right_seq = sliding_window_smooth(ear_right_seq, smoothing_window)

    anomalies: list[dict] = []
    for i, (el, er) in enumerate(zip(ear_left_seq, ear_right_seq)):
        if el is None or er is None:
            continue
        diff = abs(el - er)
        if diff > tolerance:
            anomalies.append(
                {
                    "type": "eye_asymmetry",
                    "frame_idx": i,
                    "ear_diff": round(diff, 4),
                    "severity": "low",
                    "confidence": 0.7,
                    "description": (
                        f"Frame {i}: eye asymmetry EAR diff={diff:.3f} "
                        f"(L={el:.3f}, R={er:.3f})"
                    ),
                }
            )
    return anomalies
