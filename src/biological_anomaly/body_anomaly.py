"""全身骨骼异常检测 — 借鉴 VMBench OIS 方法。

基于 MediaPipe Pose 33 关键点，检测骨段长度帧间突变和关节角度突变。
使用滑窗平滑 + 相对变化率替代绝对差分，降低误报率。
"""

from __future__ import annotations

import numpy as np

from .anomaly_rules import (
    BODY_CONSTRAINTS,
    relative_change,
    sliding_window_smooth,
)


# ---------------------------------------------------------------------------
# 骨段长度帧间突变检测
# ---------------------------------------------------------------------------


def detect_body_bone_anomalies(
    body_landmarks_seq: list[np.ndarray | None],
    constraints: dict | None = None,
    smoothing_window: int = 3,
) -> list[dict]:
    """检测全身骨段长度帧间相对突变。

    Args:
        body_landmarks_seq: 每帧 body landmarks (33, 3+)，None 表示该帧无检测
        constraints: 覆盖默认 BODY_CONSTRAINTS
        smoothing_window: 滑窗平滑窗口

    Returns:
        异常列表
    """
    c = constraints or BODY_CONSTRAINTS
    bone_segments = c["bone_segments"]
    threshold = c["bone_length_change_threshold"]
    min_valid_ratio = c["min_valid_ratio"]

    n_frames = len(body_landmarks_seq)
    if n_frames < 2:
        return []

    # 1. 逐帧计算各骨段长度
    bone_seqs: dict[str, list[float | None]] = {
        name: [] for name in bone_segments
    }
    for body in body_landmarks_seq:
        if body is None:
            for name in bone_segments:
                bone_seqs[name].append(None)
            continue
        for name, (start, end) in bone_segments.items():
            try:
                length = float(np.linalg.norm(body[start][:3] - body[end][:3]))
                bone_seqs[name].append(length)
            except (IndexError, TypeError):
                bone_seqs[name].append(None)

    # 2. 滑窗平滑
    for name in bone_seqs:
        bone_seqs[name] = sliding_window_smooth(bone_seqs[name], smoothing_window)

    # 3. 帧间相对变化率检测
    anomalies: list[dict] = []

    for name, seq in bone_seqs.items():
        valid_count = sum(1 for v in seq if v is not None)
        if valid_count / max(n_frames, 1) < min_valid_ratio:
            continue

        abnormal_frames = 0
        for i in range(1, len(seq)):
            prev, cur = seq[i - 1], seq[i]
            if prev is None or cur is None:
                continue
            change = relative_change(prev, cur)
            if change > threshold:
                abnormal_frames += 1
                anomalies.append(
                    {
                        "type": "body_bone_length_change",
                        "frame_idx": i,
                        "bone": name,
                        "change_ratio": round(change, 4),
                        "severity": _bone_severity(change, threshold),
                        "confidence": min(0.6 + change, 0.95),
                        "description": (
                            f"Frame {i}: {name} bone length changed "
                            f"{change * 100:.1f}%"
                        ),
                    }
                )

    return anomalies


# ---------------------------------------------------------------------------
# 关节角度帧间突变检测
# ---------------------------------------------------------------------------


def detect_body_angle_anomalies(
    body_landmarks_seq: list[np.ndarray | None],
    constraints: dict | None = None,
    smoothing_window: int = 3,
) -> list[dict]:
    """检测全身关节角度帧间突变。

    Args:
        body_landmarks_seq: 每帧 body landmarks (33, 3+)
        constraints: 覆盖默认 BODY_CONSTRAINTS
        smoothing_window: 滑窗平滑窗口

    Returns:
        异常列表
    """
    c = constraints or BODY_CONSTRAINTS
    joint_angles = c["joint_angles"]
    angle_threshold = c["angle_change_threshold"]
    min_valid_ratio = c["min_valid_ratio"]

    n_frames = len(body_landmarks_seq)
    if n_frames < 2:
        return []

    # 1. 逐帧计算各关节角度
    angle_seqs: dict[str, list[float | None]] = {
        name: [] for name in joint_angles
    }
    for body in body_landmarks_seq:
        if body is None:
            for name in joint_angles:
                angle_seqs[name].append(None)
            continue
        for name, (a_idx, b_idx, c_idx) in joint_angles.items():
            angle = _joint_angle_3pt(body, a_idx, b_idx, c_idx)
            angle_seqs[name].append(angle)

    # 2. 滑窗平滑
    for name in angle_seqs:
        angle_seqs[name] = sliding_window_smooth(angle_seqs[name], smoothing_window)

    # 3. 帧间绝对变化检测（角度用绝对差，VMBench 阈值 30°）
    anomalies: list[dict] = []

    for name, seq in angle_seqs.items():
        valid_count = sum(1 for v in seq if v is not None)
        if valid_count / max(n_frames, 1) < min_valid_ratio:
            continue

        for i in range(1, len(seq)):
            prev, cur = seq[i - 1], seq[i]
            if prev is None or cur is None:
                continue
            diff = abs(cur - prev)
            if diff > angle_threshold:
                anomalies.append(
                    {
                        "type": "body_angle_change",
                        "frame_idx": i,
                        "joint": name,
                        "angle_change": round(diff, 2),
                        "severity": _angle_severity(diff, angle_threshold),
                        "confidence": min(0.6 + diff / 90.0, 0.95),
                        "description": (
                            f"Frame {i}: {name} angle changed {diff:.1f}°"
                        ),
                    }
                )

    return anomalies


# ---------------------------------------------------------------------------
# 综合评分：按部位正常帧比例（VMBench OIS 风格）
# ---------------------------------------------------------------------------


def compute_body_part_scores(
    body_landmarks_seq: list[np.ndarray | None],
    constraints: dict | None = None,
    smoothing_window: int = 3,
) -> dict[str, float]:
    """计算各骨段/关节的正常帧比例得分。

    Returns:
        {part_name: normal_ratio} — 值越高越正常
    """
    c = constraints or BODY_CONSTRAINTS
    bone_segments = c["bone_segments"]
    joint_angles = c["joint_angles"]
    bone_threshold = c["bone_length_change_threshold"]
    angle_threshold = c["angle_change_threshold"]

    n_frames = len(body_landmarks_seq)
    if n_frames < 2:
        return {}

    scores: dict[str, float] = {}

    # 骨段评分
    bone_seqs: dict[str, list[float | None]] = {
        name: [] for name in bone_segments
    }
    for body in body_landmarks_seq:
        if body is None:
            for name in bone_segments:
                bone_seqs[name].append(None)
            continue
        for name, (start, end) in bone_segments.items():
            try:
                length = float(np.linalg.norm(body[start][:3] - body[end][:3]))
                bone_seqs[name].append(length)
            except (IndexError, TypeError):
                bone_seqs[name].append(None)

    for name in bone_seqs:
        bone_seqs[name] = sliding_window_smooth(bone_seqs[name], smoothing_window)

    for name, seq in bone_seqs.items():
        normal, total = _count_normal_frames(seq, bone_threshold, use_relative=True)
        if total > 0:
            scores[f"bone_{name}"] = normal / total

    # 关节评分
    angle_seqs: dict[str, list[float | None]] = {
        name: [] for name in joint_angles
    }
    for body in body_landmarks_seq:
        if body is None:
            for name in joint_angles:
                angle_seqs[name].append(None)
            continue
        for name, (a_idx, b_idx, c_idx) in joint_angles.items():
            angle = _joint_angle_3pt(body, a_idx, b_idx, c_idx)
            angle_seqs[name].append(angle)

    for name in angle_seqs:
        angle_seqs[name] = sliding_window_smooth(angle_seqs[name], smoothing_window)

    for name, seq in angle_seqs.items():
        normal, total = _count_normal_frames(
            seq, angle_threshold, use_relative=False,
        )
        if total > 0:
            scores[f"angle_{name}"] = normal / total

    return scores


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _joint_angle_3pt(
    body: np.ndarray, a_idx: int, b_idx: int, c_idx: int,
) -> float | None:
    """计算 body[a]-body[b]-body[c] 三点角度（度数）。"""
    try:
        a = body[a_idx][:3]
        b = body[b_idx][:3]
        c = body[c_idx][:3]
    except (IndexError, TypeError):
        return None
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return None
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _bone_severity(change: float, threshold: float) -> str:
    """根据变化率确定严重程度。"""
    if change > threshold * 3:
        return "high"
    if change > threshold * 1.5:
        return "medium"
    return "low"


def _angle_severity(diff: float, threshold: float) -> str:
    """根据角度变化确定严重程度。"""
    if diff > threshold * 3:
        return "high"
    if diff > threshold * 1.5:
        return "medium"
    return "low"


def _count_normal_frames(
    seq: list[float | None],
    threshold: float,
    use_relative: bool = True,
) -> tuple[int, int]:
    """统计正常帧对数量。

    Returns:
        (normal_count, total_valid_pairs)
    """
    normal = 0
    total = 0
    for i in range(1, len(seq)):
        prev, cur = seq[i - 1], seq[i]
        if prev is None or cur is None:
            continue
        total += 1
        if use_relative:
            change = relative_change(prev, cur)
        else:
            change = abs(cur - prev)
        if change <= threshold:
            normal += 1
    return normal, total
