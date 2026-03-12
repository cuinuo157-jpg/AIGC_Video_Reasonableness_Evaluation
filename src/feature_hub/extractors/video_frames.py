from __future__ import annotations

import cv2
import numpy as np


def extract_video_frames(video_path: str, device: str) -> list[np.ndarray]:
    """加载视频全部帧，返回 BGR numpy 数组列表。"""
    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames
