from __future__ import annotations

import numpy as np

from .anomaly_rules import (
    HAND_CONSTRAINTS,
    HAND_STRUCTURE_CONSTRAINTS,
    relative_change,
    sliding_window_smooth,
)


# ---------------------------------------------------------------------------
# Level 1: 运动层异常（速度突变、抖动、出现/消失）
# ---------------------------------------------------------------------------


def detect_hand_anomalies_l1(
    keypoints_seq: list[dict],
    fps: float = 30.0,
    constraints: dict | None = None,
    smoothing_window: int = 3,
) -> list[dict]:
    """Level 1: 手部运动层异常检测。

    检测项：
    - 手部突然出现/消失
    - 手腕速度突变
    - 手部抖动
    """
    c = constraints or HAND_CONSTRAINTS
    vel_threshold = c["velocity_threshold"]
    jitter_threshold = c["jitter_threshold"]
    window_size = c.get("jitter_window_size", 5)
    anomalies: list[dict] = []

    for hand_key in ("left_hand", "right_hand"):
        prev_present = False
        positions: list[tuple[int, np.ndarray | None]] = []

        for idx, kp in enumerate(keypoints_seq):
            hand = kp.get(hand_key)
            cur_present = hand is not None

            # 出现/消失检测
            if idx > 0:
                if cur_present and not prev_present:
                    anomalies.append(
                        {
                            "type": "hand_appear",
                            "hand": hand_key,
                            "frame_idx": idx,
                            "severity": "low",
                            "confidence": 0.6,
                            "description": f"Frame {idx}: {hand_key} suddenly appeared",
                        }
                    )
                elif not cur_present and prev_present:
                    anomalies.append(
                        {
                            "type": "hand_disappear",
                            "hand": hand_key,
                            "frame_idx": idx,
                            "severity": "low",
                            "confidence": 0.6,
                            "description": f"Frame {idx}: {hand_key} suddenly disappeared",
                        }
                    )

            prev_present = cur_present
            wrist = hand[0][:2] if hand is not None else None
            positions.append((idx, wrist))

        # 速度突变检测（滑窗平滑后使用相对变化率）
        velocities: list[tuple[int, float | None]] = []
        for i in range(1, len(positions)):
            _, pos1 = positions[i - 1]
            frame_idx, pos2 = positions[i]
            if pos1 is not None and pos2 is not None:
                disp = float(np.linalg.norm(pos2 - pos1))
                vel = disp * fps  # 归一化速度
                velocities.append((frame_idx, vel))
            else:
                velocities.append((frame_idx, None))

        # 对速度序列做滑窗平滑
        raw_vels = [v for _, v in velocities]
        smoothed_vels = sliding_window_smooth(raw_vels, smoothing_window)

        for i in range(1, len(velocities)):
            v1 = smoothed_vels[i - 1]
            v2 = smoothed_vels[i]
            frame_idx = velocities[i][0]
            if v1 is not None and v2 is not None:
                diff = relative_change(v1, v2)
                if diff > vel_threshold:
                    anomalies.append(
                        {
                            "type": "hand_velocity_jump",
                            "hand": hand_key,
                            "frame_idx": frame_idx,
                            "velocity_jump": round(diff, 4),
                            "severity": "medium",
                            "confidence": 0.75,
                            "description": (
                                f"Frame {frame_idx}: {hand_key} velocity jump {diff:.3f}"
                            ),
                        }
                    )

        # 抖动检测
        valid_pos = [p for _, p in positions if p is not None]
        if len(valid_pos) > window_size:
            for i in range(window_size, len(valid_pos)):
                window = np.array(valid_pos[i - window_size : i])
                jitter = float(np.std(window, axis=0).mean())
                if jitter > jitter_threshold:
                    anomalies.append(
                        {
                            "type": "hand_jitter",
                            "hand": hand_key,
                            "frame_idx": i,
                            "jitter_std": round(jitter, 5),
                            "severity": "low",
                            "confidence": 0.7,
                            "description": (
                                f"Frame {i}: {hand_key} jitter std={jitter:.4f}"
                            ),
                        }
                    )

    return anomalies


# ---------------------------------------------------------------------------
# Level 2: 结构层异常（手指融合、骨骼突变、关节角度）
# ---------------------------------------------------------------------------


def detect_hand_structure_anomalies_l2(
    hand_landmarks_seq: list[np.ndarray | None],
    hand_key: str = "hand",
    constraints: dict | None = None,
) -> list[dict]:
    """Level 2: 手部结构异常检测。

    检测项：
    - 手指融合（相邻指尖距离过近，连续多帧）
    - 骨段长度帧间突变
    - 关节角度超范围
    """
    c = constraints or HAND_STRUCTURE_CONSTRAINTS
    fingertip_idx = c["fingertip_indices"]
    min_sep = c["min_fingertip_separation"]
    fusion_frames = c["fusion_consecutive_frames"]
    bone_segments = c["bone_segments"]
    bone_tol = c.get("bone_length_change_tolerance", 0.3)
    angle_lo, angle_hi = HAND_CONSTRAINTS["joint_angle_range"]

    anomalies: list[dict] = []

    # ---- 手指融合检测 ----
    num_tips = len(fingertip_idx)
    pair_streak: dict[tuple[int, int], int] = {}
    for i in range(num_tips):
        for j in range(i + 1, num_tips):
            pair_streak[(i, j)] = 0

    for frame_idx, hand in enumerate(hand_landmarks_seq):
        if hand is None:
            for key in pair_streak:
                pair_streak[key] = 0
            continue

        tips = hand[fingertip_idx][:, :2]  # (5, 2)

        for i in range(num_tips):
            for j in range(i + 1, num_tips):
                dist = float(np.linalg.norm(tips[i] - tips[j]))
                if dist < min_sep:
                    pair_streak[(i, j)] += 1
                    if pair_streak[(i, j)] == fusion_frames:
                        anomalies.append(
                            {
                                "type": "finger_fusion",
                                "hand": hand_key,
                                "frame_idx": frame_idx,
                                "finger_pair": (
                                    fingertip_idx[i],
                                    fingertip_idx[j],
                                ),
                                "distance": round(dist, 5),
                                "severity": "high",
                                "confidence": 0.85,
                                "description": (
                                    f"Frame {frame_idx}: {hand_key} finger tips "
                                    f"{fingertip_idx[i]}&{fingertip_idx[j]} fused "
                                    f"(dist={dist:.4f}, {fusion_frames} frames)"
                                ),
                            }
                        )
                else:
                    pair_streak[(i, j)] = 0

    # ---- 骨段长度帧间突变 ----
    prev_bone_lengths: dict[str, list[float]] | None = None
    for frame_idx, hand in enumerate(hand_landmarks_seq):
        if hand is None:
            prev_bone_lengths = None
            continue

        cur_lengths: dict[str, list[float]] = {}
        for finger_name, segments in bone_segments.items():
            lengths = []
            for start, end in segments:
                length = float(np.linalg.norm(hand[start] - hand[end]))
                lengths.append(length)
            cur_lengths[finger_name] = lengths

        if prev_bone_lengths is not None:
            for finger_name in bone_segments:
                for seg_idx, (prev_l, cur_l) in enumerate(
                    zip(prev_bone_lengths[finger_name], cur_lengths[finger_name])
                ):
                    if prev_l > 1e-6:
                        change = relative_change(prev_l, cur_l)
                        if change > bone_tol:
                            anomalies.append(
                                {
                                    "type": "bone_length_change",
                                    "hand": hand_key,
                                    "frame_idx": frame_idx,
                                    "finger": finger_name,
                                    "segment_idx": seg_idx,
                                    "change_ratio": round(change, 4),
                                    "severity": "medium",
                                    "confidence": 0.7,
                                    "description": (
                                        f"Frame {frame_idx}: {hand_key} {finger_name} "
                                        f"seg{seg_idx} length changed {change * 100:.1f}%"
                                    ),
                                }
                            )

        prev_bone_lengths = cur_lengths

    # ---- 关节角度验证 ----
    for frame_idx, hand in enumerate(hand_landmarks_seq):
        if hand is None:
            continue
        for finger_name, segments in bone_segments.items():
            if len(segments) < 2:
                continue
            for seg_i in range(len(segments) - 1):
                _, mid = segments[seg_i]
                _, end = segments[seg_i + 1]
                start_pt = segments[seg_i][0]
                angle = _joint_angle(hand[start_pt], hand[mid], hand[end])
                if angle is not None and (angle < angle_lo or angle > angle_hi):
                    anomalies.append(
                        {
                            "type": "impossible_joint_angle",
                            "hand": hand_key,
                            "frame_idx": frame_idx,
                            "finger": finger_name,
                            "angle": round(angle, 2),
                            "severity": "medium",
                            "confidence": 0.8,
                            "description": (
                                f"Frame {frame_idx}: {hand_key} {finger_name} "
                                f"joint angle {angle:.1f} out of [{angle_lo}, {angle_hi}]"
                            ),
                        }
                    )

    return anomalies


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    """计算 a-b-c 三点形成的角度（度数）。"""
    ba = a[:3] - b[:3]
    bc = c[:3] - b[:3]
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return None
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


# ---------------------------------------------------------------------------
# Level 2: 手指伸展计数（检测多指/缺指）
# ---------------------------------------------------------------------------


def detect_finger_count_anomalies(
    hand_landmarks_seq: list[np.ndarray | None],
    hand_key: str = "hand",
    expected_count: int = 5,
) -> list[dict]:
    """Level 2: 基于关键点拓扑判断手指伸展数量。

    通过比较指尖与 MCP 关节到手腕的距离来判断手指是否伸展。
    拇指特殊处理：使用 x 坐标与 MCP 的横向偏移判断。

    Args:
        hand_landmarks_seq: 手部关键点序列 (21, 3+)
        hand_key: "left_hand" 或 "right_hand"
        expected_count: 期望手指数量

    Returns:
        异常列表（手指数 != expected_count 的帧）
    """
    fingertip_idx = [4, 8, 12, 16, 20]
    finger_pip_idx = [3, 6, 10, 14, 18]  # PIP 关节（用于伸展判断）

    anomalies: list[dict] = []
    counts: list[int | None] = []

    for frame_idx, hand in enumerate(hand_landmarks_seq):
        if hand is None:
            counts.append(None)
            continue

        extended = 0

        # 拇指：比较指尖与 IP 关节的 x 坐标距离
        # 判断拇指是否向外展开
        thumb_tip = hand[4][:2]
        thumb_ip = hand[3][:2]
        thumb_mcp = hand[2][:2]
        # 拇指伸展 = 指尖比 IP 关节更远离手掌中心
        if np.linalg.norm(thumb_tip - thumb_mcp) > np.linalg.norm(
            thumb_ip - thumb_mcp
        ):
            extended += 1

        # 其余四指：指尖比 PIP 关节更远离手腕
        wrist = hand[0][:2]
        for tip_i, pip_i in zip(fingertip_idx[1:], finger_pip_idx[1:]):
            tip_dist = np.linalg.norm(hand[tip_i][:2] - wrist)
            pip_dist = np.linalg.norm(hand[pip_i][:2] - wrist)
            if tip_dist > pip_dist:
                extended += 1

        counts.append(extended)

        if extended != expected_count:
            severity = "high" if abs(extended - expected_count) >= 2 else "medium"
            anomalies.append(
                {
                    "type": "finger_count_abnormal",
                    "hand": hand_key,
                    "frame_idx": frame_idx,
                    "finger_count": extended,
                    "expected": expected_count,
                    "severity": severity,
                    "confidence": 0.7,
                    "description": (
                        f"Frame {frame_idx}: {hand_key} has {extended} "
                        f"extended fingers (expected {expected_count})"
                    ),
                }
            )

    return anomalies
