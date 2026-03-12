from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

from .anomaly_rules import MOUTH_CONSTRAINTS, relative_change, sliding_window_smooth


def detect_mouth_anomalies_l1(
    mar_sequence: list[float | None],
    fps: float = 30.0,
    constraints: dict | None = None,
    smoothing_window: int = 3,
) -> list[dict]:
    """Level 1: 基于 MAR 序列检测嘴部异常。

    检测项：
    - MAR 帧间跳跃（嘴型不连续）
    - 持续张嘴超时
    """
    c = constraints or MOUTH_CONSTRAINTS
    jump_threshold = c["mar_jump_threshold"]
    open_threshold = c["mar_sustained_open_threshold"]
    max_open_s = c["max_open_duration_s"]
    anomalies: list[dict] = []

    # 滑窗平滑降低噪声误报
    mar_sequence = sliding_window_smooth(mar_sequence, smoothing_window)

    # MAR 跳跃检测（使用相对变化率）
    for i in range(1, len(mar_sequence)):
        prev, cur = mar_sequence[i - 1], mar_sequence[i]
        if prev is None or cur is None:
            continue
        diff = relative_change(prev, cur)
        if diff > jump_threshold:
            anomalies.append(
                {
                    "type": "mouth_discontinuity",
                    "frame_idx": i,
                    "mar_jump": round(diff, 4),
                    "severity": "medium",
                    "confidence": 0.8,
                    "description": f"Frame {i}: MAR jump {diff:.3f}",
                }
            )

    # 持续张嘴检测
    open_start = -1
    for i, mar in enumerate(mar_sequence):
        if mar is None:
            open_start = -1
            continue
        if mar > open_threshold:
            if open_start == -1:
                open_start = i
        else:
            if open_start != -1:
                duration_s = (i - open_start) / fps
                if duration_s > max_open_s:
                    anomalies.append(
                        {
                            "type": "prolonged_mouth_opening",
                            "frame_idx": open_start,
                            "duration_s": round(duration_s, 2),
                            "severity": "low",
                            "confidence": 0.7,
                            "description": (
                                f"Frame {open_start}: mouth open for {duration_s:.1f}s"
                            ),
                        }
                    )
                open_start = -1

    return anomalies


def detect_mouth_temporal_anomalies_l2(
    face_landmarks_seq: list[np.ndarray | None],
    fps: float = 30.0,
    constraints: dict | None = None,
) -> list[dict]:
    """Level 2: 基于嘴内 landmark 时序一致性检测结构异常。

    检测项：
    - 嘴内区域面积帧间突变（舌头/牙齿消失/出现）
    - 嘴部 landmark 位置跳跃
    """
    c = constraints or MOUTH_CONSTRAINTS
    inner_lip_indices = c["inner_lip_indices"]
    area_threshold = c["area_change_threshold"]
    stability_threshold = c["stability_threshold"]
    anomalies: list[dict] = []

    prev_area: float | None = None
    prev_centroid: np.ndarray | None = None

    for i, face in enumerate(face_landmarks_seq):
        if face is None:
            prev_area = None
            prev_centroid = None
            continue

        # 提取嘴内轮廓点
        try:
            lip_pts = face[inner_lip_indices]  # (N, 3) 取 x, y
            lip_2d = lip_pts[:, :2]
        except (IndexError, TypeError):
            prev_area = None
            prev_centroid = None
            continue

        # 计算内唇多边形面积（Shoelace 公式）
        area = _polygon_area(lip_2d)
        centroid = lip_2d.mean(axis=0)

        if prev_area is not None and prev_area > 1e-8:
            area_ratio = relative_change(prev_area, area)
            if area_ratio > area_threshold:
                anomalies.append(
                    {
                        "type": "mouth_area_sudden_change",
                        "frame_idx": i,
                        "area_change_ratio": round(area_ratio, 4),
                        "severity": "medium",
                        "confidence": 0.75,
                        "description": (
                            f"Frame {i}: inner mouth area changed by "
                            f"{area_ratio * 100:.1f}%"
                        ),
                    }
                )

        if prev_centroid is not None:
            shift = float(np.linalg.norm(centroid - prev_centroid))
            if shift > stability_threshold:
                anomalies.append(
                    {
                        "type": "mouth_landmark_jump",
                        "frame_idx": i,
                        "shift": round(shift, 5),
                        "severity": "low",
                        "confidence": 0.7,
                        "description": (
                            f"Frame {i}: mouth centroid shift {shift:.4f}"
                        ),
                    }
                )

        prev_area = area
        prev_centroid = centroid

    return anomalies


def _polygon_area(pts: np.ndarray) -> float:
    """Shoelace 公式计算 2D 多边形面积。"""
    n = len(pts)
    if n < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# ---------------------------------------------------------------------------
# 嘴内颜色直方图突变检测（牙齿/舌头/残渣消失）
# ---------------------------------------------------------------------------


def detect_mouth_color_anomalies(
    frames: list[np.ndarray],
    face_landmarks_seq: list[np.ndarray | None],
    constraints: dict | None = None,
) -> list[dict]:
    """基于嘴内 ROI 颜色直方图帧间相关性检测突变。

    当舌头、牙齿或嘴上残渣突然消失/出现时，嘴内区域的颜色分布
    会发生剧烈变化，表现为直方图相关性骤降。

    Args:
        frames: 视频帧列表 (BGR)
        face_landmarks_seq: 面部 landmarks 序列 (468, 3)
        constraints: 约束参数

    Returns:
        异常列表
    """
    if cv2 is None:
        return []
    c = constraints or MOUTH_CONSTRAINTS
    inner_lip_indices = c["inner_lip_indices"]
    hist_threshold = c.get("histogram_correlation_threshold", 0.7)
    anomalies: list[dict] = []

    prev_hist: np.ndarray | None = None

    for i, (frame, face) in enumerate(zip(frames, face_landmarks_seq)):
        if face is None:
            prev_hist = None
            continue

        # 提取嘴内多边形 ROI
        try:
            lip_pts = face[inner_lip_indices][:, :2]
        except (IndexError, TypeError):
            prev_hist = None
            continue

        h, w = frame.shape[:2]
        # 转为像素坐标
        pts_px = (lip_pts * np.array([w, h])).astype(np.int32)

        # 创建嘴内掩膜
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_px], 255)

        # 检查掩膜是否有效
        if cv2.countNonZero(mask) < 10:
            prev_hist = None
            continue

        # 计算 HSV 颜色直方图
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256],
        )
        cv2.normalize(hist, hist)

        if prev_hist is not None:
            # 直方图相关性（1.0 = 完全相同，越低越不同）
            correlation = cv2.compareHist(
                prev_hist, hist, cv2.HISTCMP_CORREL,
            )
            if correlation < hist_threshold:
                severity = "high" if correlation < 0.3 else "medium"
                anomalies.append(
                    {
                        "type": "mouth_color_sudden_change",
                        "frame_idx": i,
                        "correlation": round(float(correlation), 4),
                        "severity": severity,
                        "confidence": min(0.6 + (1.0 - correlation) * 0.5, 0.95),
                        "description": (
                            f"Frame {i}: mouth interior color histogram "
                            f"correlation dropped to {correlation:.3f}"
                        ),
                    }
                )

        prev_hist = hist

    return anomalies
