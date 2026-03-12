"""生物特征异常检测约束常量与工具函数。"""

from __future__ import annotations

import numpy as np


# ================================================================
# 通用时序工具
# ================================================================


def sliding_window_smooth(
    seq: list[float | None], window: int = 3,
) -> list[float | None]:
    """对可空序列进行滑动窗口均值平滑。

    None 值保持不变；窗口内 None 被跳过。
    """
    if window < 2:
        return seq
    half = window // 2
    n = len(seq)
    out: list[float | None] = []
    for i in range(n):
        if seq[i] is None:
            out.append(None)
            continue
        vals = []
        for j in range(max(0, i - half), min(n, i + half + 1)):
            if seq[j] is not None:
                vals.append(seq[j])
        out.append(float(np.mean(vals)) if vals else seq[i])
    return out


def relative_change(prev: float, cur: float) -> float:
    """计算相对变化率 |cur - prev| / max(|prev|, eps)。"""
    denom = max(abs(prev), 1e-8)
    return abs(cur - prev) / denom

# ---------- 眼部约束 ----------
EYE_CONSTRAINTS = {
    "ear_blink_threshold": 0.21,
    "max_no_blink_frames": 90,
    "symmetry_tolerance": 0.15,
}

# ---------- 手部约束（L1 + L2） ----------
HAND_CONSTRAINTS = {
    "finger_count": 5,
    "joint_angle_range": (0, 180),
    "thumb_angle_range": (-30, 130),
    "bone_length_ratio_tolerance": 0.15,
    "velocity_threshold": 0.3,
    "jitter_threshold": 0.05,
    "jitter_window_size": 5,
}

HAND_STRUCTURE_CONSTRAINTS = {
    # MediaPipe 手部 21 关键点中的指尖索引
    "fingertip_indices": [4, 8, 12, 16, 20],
    # 各指 MCP 关节索引（用于判断手指是否伸展）
    "finger_mcp_indices": [2, 5, 9, 13, 17],
    # 指尖最小间距（归一化坐标），低于此值视为融合
    "min_fingertip_separation": 0.02,
    # 连续帧数阈值，连续 N 帧低于间距才确认融合
    "fusion_consecutive_frames": 3,
    # 各手指骨段定义 (起点索引, 终点索引)
    "bone_segments": {
        "thumb": [(1, 2), (2, 3), (3, 4)],
        "index": [(5, 6), (6, 7), (7, 8)],
        "middle": [(9, 10), (10, 11), (11, 12)],
        "ring": [(13, 14), (14, 15), (15, 16)],
        "pinky": [(17, 18), (18, 19), (19, 20)],
    },
    # 骨段长度帧间变化容忍度（比值）
    "bone_length_change_tolerance": 0.3,
}

# ---------- 嘴部约束（L1 + L2） ----------
MOUTH_CONSTRAINTS = {
    "mar_jump_threshold": 0.3,
    "mar_sustained_open_threshold": 0.5,
    "max_open_duration_s": 3.0,
    # MediaPipe 468 面部网格中的内唇轮廓索引
    "inner_lip_indices": [
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    ],
    # 嘴内区域面积帧间突变阈值（比率）
    "area_change_threshold": 0.5,
    # landmark 稳定性阈值（归一化坐标跳跃）
    "stability_threshold": 0.05,
    # 嘴内颜色直方图帧间相关性突降阈值
    "histogram_correlation_threshold": 0.7,
}

# ---------- 全身骨骼约束（借鉴 VMBench OIS） ----------
# MediaPipe Pose 33 点中的关键索引 (COCO 近似映射):
#   0=nose, 11=L_shoulder, 12=R_shoulder, 13=L_elbow, 14=R_elbow,
#   15=L_wrist, 16=R_wrist, 23=L_hip, 24=R_hip, 25=L_knee,
#   26=R_knee, 27=L_ankle, 28=R_ankle
BODY_CONSTRAINTS = {
    # 12 个骨段: (起点索引, 终点索引, 名称)
    "bone_segments": {
        "torso": (11, 23),
        "left_upper_arm": (11, 13),
        "right_upper_arm": (12, 14),
        "left_forearm": (13, 15),
        "right_forearm": (14, 16),
        "left_thigh": (23, 25),
        "right_thigh": (24, 26),
        "left_shin": (25, 27),
        "right_shin": (26, 28),
        "shoulder_width": (11, 12),
        "hip_width": (23, 24),
        "neck": (0, 11),
    },
    # 8 个关节: (端点A, 关节点, 端点B, 名称)
    "joint_angles": {
        "left_elbow": (11, 13, 15),
        "right_elbow": (12, 14, 16),
        "left_shoulder": (13, 11, 23),
        "right_shoulder": (14, 12, 24),
        "left_hip": (11, 23, 25),
        "right_hip": (12, 24, 26),
        "left_knee": (23, 25, 27),
        "right_knee": (24, 26, 28),
    },
    # 骨段长度帧间相对变化阈值 (VMBench 用 45%)
    "bone_length_change_threshold": 0.45,
    # 关节角度帧间绝对变化阈值 (VMBench 用 30°)
    "angle_change_threshold": 30.0,
    # 最小有效帧比例 (低于此比例视为数据不足，不判异常)
    "min_valid_ratio": 0.5,
}
