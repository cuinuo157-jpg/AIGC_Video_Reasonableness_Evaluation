from __future__ import annotations

import cv2
import numpy as np

from src.feature_hub.hub import VideoProcessingConfig


def _resize_frame(frame: np.ndarray, max_side: int | None) -> np.ndarray:
    if not max_side or max_side <= 0:
        return frame
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame
    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _subsample_uniform(
    frames: list[np.ndarray],
    max_frames: int | None,
) -> list[np.ndarray]:
    if max_frames is None or max_frames <= 0 or len(frames) <= max_frames:
        return frames
    indices = np.linspace(0, len(frames) - 1, num=max_frames, dtype=int)
    return [frames[idx] for idx in indices.tolist()]


def load_video_frames(
    video_path: str,
    video_config: VideoProcessingConfig | None = None,
) -> list[np.ndarray]:
    """按统一配置加载视频帧，返回 BGR numpy 数组列表。"""
    cfg = video_config or VideoProcessingConfig()
    sample_stride = max(1, int(cfg.sample_stride))
    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_stride == 0:
            frames.append(_resize_frame(frame, cfg.max_side))
        frame_idx += 1
    cap.release()
    return _subsample_uniform(frames, cfg.max_frames)


def extract_video_frames(
    video_path: str,
    device: str,
    hub: object | None = None,
) -> list[np.ndarray]:
    video_config = getattr(hub, "video_config", None)
    return load_video_frames(video_path, video_config=video_config)
