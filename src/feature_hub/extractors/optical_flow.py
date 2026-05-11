from __future__ import annotations

import cv2
import numpy as np

from .video_frames import load_video_frames


def _compute_flows(
    frames: list[np.ndarray], device: str
) -> list[tuple[np.ndarray, np.ndarray]]:
    flows = []
    for i in range(len(frames) - 1):
        gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flows.append((flow[..., 0], flow[..., 1]))
    return flows


def extract_optical_flow(
    video_path: str,
    device: str,
    hub: object | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if hub is not None:
        frames = hub.get("video_frames")
    else:
        frames = load_video_frames(video_path)
    if len(frames) < 2:
        return []
    return _compute_flows(frames, device)
