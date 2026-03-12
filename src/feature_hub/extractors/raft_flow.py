"""RAFT / TV-L1 / Farneback 多方法光流提取器。

复用 aux_motion_intensity.flow_predictor.SimpleRAFT 的成熟实现，
RAFT 不可用时自动降级为 Farneback。
"""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _load_frames_rgb(video_path: str) -> list[np.ndarray]:
    """加载视频帧为 RGB 格式。"""
    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def extract_raft_flow(
    video_path: str,
    device: str,
    hub: Any = None,
    method: str = "raft",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """使用 RAFT/TV-L1/Farneback 提取光流序列。

    返回格式与 optical_flow extractor 一致:
      list[tuple[flow_x (H,W), flow_y (H,W)]]

    Args:
        video_path: 视频文件路径。
        device: 推理设备 ("cuda" / "cpu")。
        hub: FeatureHub 实例 (未使用，保留接口兼容)。
        method: 光流方法 ("raft" / "tvl1" / "farneback")。
    """
    frames = _load_frames_rgb(video_path)
    if len(frames) < 2:
        return []

    try:
        from src.aux_motion_intensity.flow_predictor import SimpleRAFT

        predictor = SimpleRAFT(device=device, method=method)
    except (ImportError, Exception) as e:
        logger.warning("SimpleRAFT 初始化失败 (%s)，降级为 Farneback", e)
        predictor = None

    flows: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(frames) - 1):
        if predictor is not None:
            # SimpleRAFT.predict_flow 返回 (2, H, W)
            flow_2hw = predictor.predict_flow(frames[i], frames[i + 1])
            flows.append((flow_2hw[0], flow_2hw[1]))
        else:
            # Farneback 降级
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flows.append((flow[..., 0], flow[..., 1]))
    return flows
