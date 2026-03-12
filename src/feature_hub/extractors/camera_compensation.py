"""相机运动补偿特征提取器。

使用 SIFT/ORB + Homography + RANSAC 分离相机运动与目标运动，
从原始光流中提取残差光流。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraCompensationResult:
    """相机补偿结果。"""
    residual_flows: list[np.ndarray] = field(default_factory=list)
    homographies: list[np.ndarray | None] = field(default_factory=list)
    camera_magnitude: float = 0.0


def extract_camera_compensation(
    video_path: str,
    device: str,
    hub: Any = None,
) -> CameraCompensationResult:
    """提取相机补偿后的残差光流。

    依赖 hub.get("optical_flow") 获取原始光流缓存结果，
    再逐帧应用 Homography 估计分离相机运动。

    Args:
        video_path: 视频文件路径。
        device: 推理设备。
        hub: FeatureHub 实例 (用于获取 optical_flow 缓存)。

    Returns:
        CameraCompensationResult 包含残差光流列表、Homography 列表和相机运动幅度。
    """
    if hub is None:
        logger.warning("camera_compensation 需要 hub 参数以获取光流缓存")
        return CameraCompensationResult()

    # 获取原始光流
    raw_flows = hub.get("optical_flow")
    if not raw_flows:
        return CameraCompensationResult()

    # 加载视频帧 (用于特征匹配)
    frames = hub.get("video_frames")
    if not frames or len(frames) < 2:
        return CameraCompensationResult()

    try:
        from src.aux_motion_intensity.camera_compensation import CameraCompensator

        compensator = CameraCompensator(feature="SIFT", temporal_smooth=True)
    except ImportError:
        logger.warning("CameraCompensator 不可用，使用简化补偿")
        compensator = None

    residual_flows: list[np.ndarray] = []
    homographies: list[np.ndarray | None] = []
    camera_magnitudes: list[float] = []

    for i, (flow_x, flow_y) in enumerate(raw_flows):
        if i + 1 >= len(frames):
            break
        flow_hw2 = np.stack([flow_x, flow_y], axis=-1)  # (H, W, 2)

        if compensator is not None:
            # 使用成熟的相机补偿实现
            result = compensator.compensate(flow_hw2, frames[i], frames[i + 1])
            residual = result["residual_flow"]
            H = result["homography"]
            cam_flow = result["camera_flow"]
        else:
            # 简化方案: 全局中值作为相机运动估计
            median_flow = np.median(flow_hw2, axis=(0, 1))
            residual = flow_hw2 - median_flow
            H = None
            cam_flow = np.broadcast_to(median_flow, flow_hw2.shape)

        residual_flows.append(residual)
        homographies.append(H)
        camera_magnitudes.append(float(np.mean(np.linalg.norm(cam_flow, axis=-1))))

    avg_camera_mag = float(np.mean(camera_magnitudes)) if camera_magnitudes else 0.0

    return CameraCompensationResult(
        residual_flows=residual_flows,
        homographies=homographies,
        camera_magnitude=avg_camera_mag,
    )
