"""相机运动补偿特征提取器。

自包含实现：
  1. 使用 SIFT/ORB 做相邻帧特征匹配
  2. 用 RANSAC 估计 Homography
  3. 将全局相机位移从原始光流中扣除，得到残差光流

当特征点不足或 Homography 估计失败时，降级为全局中值光流补偿。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MIN_MATCH_COUNT = 8
_RANSAC_REPROJ_THRESHOLD = 3.0


@dataclass
class CameraCompensationResult:
    """相机补偿结果。"""

    residual_flows: list[np.ndarray] = field(default_factory=list)
    homographies: list[np.ndarray | None] = field(default_factory=list)
    camera_magnitude: float = 0.0


def _build_feature_detector() -> tuple[Any, Any, str]:
    """优先使用 SIFT，不可用时退回 ORB。"""
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create()
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        return detector, matcher, "SIFT"

    detector = cv2.ORB_create(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    return detector, matcher, "ORB"


def _estimate_homography(frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray | None:
    """估计从 frame_a 到 frame_b 的单应矩阵。"""
    detector, matcher, feature_name = _build_feature_detector()
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    keypoints_a, descriptors_a = detector.detectAndCompute(gray_a, None)
    keypoints_b, descriptors_b = detector.detectAndCompute(gray_b, None)
    if descriptors_a is None or descriptors_b is None:
        return None
    if len(keypoints_a) < _MIN_MATCH_COUNT or len(keypoints_b) < _MIN_MATCH_COUNT:
        return None

    try:
        knn_matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    except cv2.error:
        return None

    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < _MIN_MATCH_COUNT:
        logger.debug("Camera compensation matched only %s features with %s", len(good_matches), feature_name)
        return None

    src_pts = np.float32([keypoints_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography, _ = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        ransacReprojThreshold=_RANSAC_REPROJ_THRESHOLD,
    )
    return homography


def _camera_flow_from_homography(homography: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """将单应矩阵转换为逐像素相机光流。"""
    height, width = shape
    grid_y, grid_x = np.indices((height, width), dtype=np.float32)
    points = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(points, homography).reshape(height, width, 2)
    original = np.stack([grid_x, grid_y], axis=-1)
    return warped - original


def _median_flow_compensation(flow_hw2: np.ndarray) -> tuple[np.ndarray, np.ndarray, None]:
    """退化为全局中值光流补偿。"""
    median_flow = np.median(flow_hw2, axis=(0, 1))
    camera_flow = np.broadcast_to(median_flow, flow_hw2.shape).copy()
    residual = flow_hw2 - camera_flow
    return residual, camera_flow, None


def extract_camera_compensation(
    video_path: str,
    device: str,
    hub: Any = None,
) -> CameraCompensationResult:
    """提取相机补偿后的残差光流。"""
    if hub is None:
        logger.warning("camera_compensation 需要 hub 参数以获取光流缓存")
        return CameraCompensationResult()

    raw_flows = hub.get("optical_flow")
    if not raw_flows:
        return CameraCompensationResult()

    frames = hub.get("video_frames")
    if not frames or len(frames) < 2:
        return CameraCompensationResult()

    residual_flows: list[np.ndarray] = []
    homographies: list[np.ndarray | None] = []
    camera_magnitudes: list[float] = []
    used_homography = False
    fallback_count = 0

    for i, (flow_x, flow_y) in enumerate(raw_flows):
        if i + 1 >= len(frames):
            break

        flow_hw2 = np.stack([flow_x, flow_y], axis=-1).astype(np.float32, copy=False)
        homography = _estimate_homography(frames[i], frames[i + 1])

        if homography is not None:
            camera_flow = _camera_flow_from_homography(homography, flow_hw2.shape[:2])
            residual = flow_hw2 - camera_flow
            used_homography = True
        else:
            residual, camera_flow, homography = _median_flow_compensation(flow_hw2)
            fallback_count += 1

        residual_flows.append(residual)
        homographies.append(homography)
        camera_magnitudes.append(float(np.mean(np.linalg.norm(camera_flow, axis=-1))))

    if fallback_count and not used_homography:
        logger.warning("Camera compensation fallback to median flow for all frames")
    elif fallback_count:
        logger.info("Camera compensation partially fell back to median flow: %s frames", fallback_count)

    avg_camera_mag = float(np.mean(camera_magnitudes)) if camera_magnitudes else 0.0
    return CameraCompensationResult(
        residual_flows=residual_flows,
        homographies=homographies,
        camera_magnitude=avg_camera_mag,
    )
