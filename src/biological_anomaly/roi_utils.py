from __future__ import annotations

import cv2
import numpy as np

from .anomaly_rules import MOUTH_CONSTRAINTS


def crop_hand_roi(
    frame: np.ndarray,
    hand_landmarks: np.ndarray,
    padding: float = 0.2,
) -> np.ndarray:
    """从帧中裁剪手部 ROI 区域。

    Args:
        frame: BGR 图像 (H, W, 3)
        hand_landmarks: 手部关键点 (21, 3)，归一化坐标
        padding: 边界扩展比例

    Returns:
        裁剪后的 ROI 图像
    """
    h, w = frame.shape[:2]
    pts = hand_landmarks[:, :2]
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    # 像素坐标 + padding
    pad_x = (x_max - x_min) * padding
    pad_y = (y_max - y_min) * padding
    x1 = max(0, int((x_min - pad_x) * w))
    y1 = max(0, int((y_min - pad_y) * h))
    x2 = min(w, int((x_max + pad_x) * w))
    y2 = min(h, int((y_max + pad_y) * h))

    if x2 <= x1 or y2 <= y1:
        return frame  # fallback
    return frame[y1:y2, x1:x2].copy()


def crop_mouth_roi(
    frame: np.ndarray,
    face_landmarks: np.ndarray,
    padding: float = 0.3,
) -> np.ndarray:
    """从帧中裁剪嘴部 ROI 区域。

    Args:
        frame: BGR 图像 (H, W, 3)
        face_landmarks: 面部关键点 (468, 3)，归一化坐标
        padding: 边界扩展比例

    Returns:
        裁剪后的 ROI 图像
    """
    h, w = frame.shape[:2]
    mouth_indices = MOUTH_CONSTRAINTS["inner_lip_indices"]
    pts = face_landmarks[mouth_indices][:, :2]
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    pad_x = (x_max - x_min) * padding
    pad_y = (y_max - y_min) * padding
    x1 = max(0, int((x_min - pad_x) * w))
    y1 = max(0, int((y_min - pad_y) * h))
    x2 = min(w, int((x_max + pad_x) * w))
    y2 = min(h, int((y_max + pad_y) * h))

    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2].copy()


def extract_suspicious_rois(
    frames: list[np.ndarray],
    keypoints_seq: list[dict],
    suspicious_frames: list[dict],
    max_crops: int = 8,
) -> list[dict]:
    """从疑似异常帧中提取 ROI 裁剪图像。

    Args:
        frames: 视频全部帧（BGR）
        keypoints_seq: 对应的关键点序列
        suspicious_frames: 异常信息列表，每个含 frame_idx 和 type
        max_crops: 最大裁剪数

    Returns:
        ROI 信息列表：[{"frame_idx", "region_type", "crop", "anomaly_info"}, ...]
    """
    rois: list[dict] = []

    for info in suspicious_frames[:max_crops]:
        idx = info.get("frame_idx", 0)
        if idx < 0 or idx >= len(frames) or idx >= len(keypoints_seq):
            continue

        frame = frames[idx]
        kp = keypoints_seq[idx]
        anomaly_type = info.get("type", "")

        # 根据异常类型决定裁剪区域
        if "hand" in anomaly_type or "finger" in anomaly_type or "bone" in anomaly_type:
            hand_key = info.get("hand", "left_hand")
            hand_lm = kp.get(hand_key)
            if hand_lm is not None:
                crop = crop_hand_roi(frame, hand_lm)
                rois.append(
                    {
                        "frame_idx": idx,
                        "region_type": "hand",
                        "crop": crop,
                        "anomaly_info": info,
                    }
                )
        elif "mouth" in anomaly_type:
            face_lm = kp.get("face")
            if face_lm is not None:
                crop = crop_mouth_roi(frame, face_lm)
                rois.append(
                    {
                        "frame_idx": idx,
                        "region_type": "mouth",
                        "crop": crop,
                        "anomaly_info": info,
                    }
                )

    return rois
