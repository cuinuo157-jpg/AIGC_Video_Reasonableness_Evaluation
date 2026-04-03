"""MediaPipe Iris 瞳孔追踪提取器（基于已缓存 face landmarks）。

设计说明：
- 不重复调用 MediaPipe，直接复用 hub.get("keypoints") 的面部 landmarks；
- 若 face landmarks 包含 refine_face_landmarks 输出的 478 点，则可直接读取虹膜点；
- 不可用时返回空特征（每帧 None），保证流水线鲁棒。
"""
from __future__ import annotations

from typing import Any

import numpy as np

# MediaPipe Face Mesh (refine_face_landmarks=True) 虹膜索引
_LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
_RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

# 眼角索引（用于归一化尺度）
_LEFT_EYE_CORNERS = (33, 133)
_RIGHT_EYE_CORNERS = (362, 263)


def _safe_center(face: np.ndarray, indices: list[int]) -> np.ndarray | None:
    if face is None or len(face) <= max(indices):
        return None
    pts = face[indices, :2]
    return pts.mean(axis=0).astype(np.float32)


def _safe_radius(face: np.ndarray, center: np.ndarray, indices: list[int]) -> float | None:
    if face is None or center is None or len(face) <= max(indices):
        return None
    pts = face[indices, :2]
    d = np.linalg.norm(pts - center[None, :], axis=1)
    if d.size == 0:
        return None
    return float(np.mean(d))


def _eye_width(face: np.ndarray, corners: tuple[int, int]) -> float | None:
    if face is None or len(face) <= max(corners):
        return None
    a, b = corners
    return float(np.linalg.norm(face[a, :2] - face[b, :2]))


def extract_iris_tracking(
    video_path: str,
    device: str,
    hub: Any = None,
) -> list[dict]:
    """提取每帧瞳孔追踪特征。

    Returns:
        list[dict], 每帧包含：
        - left_pupil_center / right_pupil_center: np.ndarray([x, y]) | None
        - left_pupil_radius / right_pupil_radius: float | None
        - interpupil_distance: float | None（双瞳距，归一化坐标）
        - left_pupil_radius_norm / right_pupil_radius_norm: float | None（按眼宽归一化）
    """
    if hub is None:
        return []

    keypoints_seq = hub.get("keypoints")
    results: list[dict] = []

    for idx, kp in enumerate(keypoints_seq):
        face = kp.get("face")
        left_center = _safe_center(face, _LEFT_IRIS_INDICES) if face is not None else None
        right_center = _safe_center(face, _RIGHT_IRIS_INDICES) if face is not None else None

        left_radius = _safe_radius(face, left_center, _LEFT_IRIS_INDICES) if left_center is not None else None
        right_radius = _safe_radius(face, right_center, _RIGHT_IRIS_INDICES) if right_center is not None else None

        ipd = None
        if left_center is not None and right_center is not None:
            ipd = float(np.linalg.norm(left_center - right_center))

        left_eye_w = _eye_width(face, _LEFT_EYE_CORNERS) if face is not None else None
        right_eye_w = _eye_width(face, _RIGHT_EYE_CORNERS) if face is not None else None

        left_radius_norm = None
        if left_radius is not None and left_eye_w is not None and left_eye_w > 1e-8:
            left_radius_norm = float(left_radius / left_eye_w)

        right_radius_norm = None
        if right_radius is not None and right_eye_w is not None and right_eye_w > 1e-8:
            right_radius_norm = float(right_radius / right_eye_w)

        results.append(
            {
                "frame_idx": idx,
                "left_pupil_center": left_center,
                "right_pupil_center": right_center,
                "left_pupil_radius": left_radius,
                "right_pupil_radius": right_radius,
                "interpupil_distance": ipd,
                "left_pupil_radius_norm": left_radius_norm,
                "right_pupil_radius_norm": right_radius_norm,
            }
        )

    return results
