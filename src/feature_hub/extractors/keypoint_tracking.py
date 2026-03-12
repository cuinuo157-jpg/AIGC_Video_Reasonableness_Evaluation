"""关键点轨迹提取器 — 从 MediaPipe 关键点序列中提取身体关节时序轨迹。

轻量实现，不依赖 SAM2/DINO/Co-Tracker。
为 physics_consistency 重力检测提供轨迹数据。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe Pose 33-point 中用于轨迹追踪的关键关节索引
_TRACKED_JOINTS = {
    "nose": 0,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_hip": 23,
    "right_hip": 24,
}


def extract_keypoint_trajectories(
    video_path: str,
    device: str,
    hub: Any = None,
) -> list[np.ndarray]:
    """提取关键关节的时序轨迹。

    从 hub.get("keypoints") 获取 MediaPipe 关键点序列，
    提取指定关节在所有帧中的 (x, y) 位置序列。

    Args:
        video_path: 视频文件路径 (未直接使用，由 hub 管理)。
        device: 推理设备 (未直接使用)。
        hub: FeatureHub 实例。

    Returns:
        轨迹列表，每条轨迹为 np.ndarray shape (T, 2)，
        其中 T 为有效帧数。无效帧位置为 (nan, nan)。
    """
    if hub is None:
        logger.warning("keypoint_tracking 需要 hub 参数以获取关键点缓存")
        return []

    keypoints_seq = hub.get("keypoints")
    if not keypoints_seq:
        return []

    n_frames = len(keypoints_seq)
    joint_names = list(_TRACKED_JOINTS.keys())
    joint_indices = list(_TRACKED_JOINTS.values())

    # 初始化轨迹数组 (每个关节一条轨迹)
    trajectories: list[np.ndarray] = []
    for _ in joint_indices:
        trajectories.append(np.full((n_frames, 2), np.nan, dtype=np.float32))

    for frame_idx, kp_dict in enumerate(keypoints_seq):
        body = kp_dict.get("body")
        if body is None:
            continue
        for j, joint_idx in enumerate(joint_indices):
            if joint_idx < len(body):
                # MediaPipe 归一化坐标 (x, y)
                trajectories[j][frame_idx, 0] = body[joint_idx][0]
                trajectories[j][frame_idx, 1] = body[joint_idx][1]

    # 过滤掉全为 nan 的轨迹
    valid = [t for t in trajectories if not np.all(np.isnan(t))]
    return valid
