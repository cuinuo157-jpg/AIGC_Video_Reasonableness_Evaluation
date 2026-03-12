"""AU 特征提取器 — 包装 Py-Feat AUExtractor 以便跨维度共享。

将 expression_naturalness 的 AU 提取能力注册到 FeatureHub，
避免多个维度分别加载 Py-Feat 模型。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def extract_au_features(
    video_path: str,
    device: str,
    hub: Any = None,
) -> list[dict[str, float]]:
    """提取视频逐帧 AU 强度特征。

    依赖 hub.get("video_frames") 获取帧数据，
    使用 Py-Feat Detector 提取每帧的 Action Unit 强度。

    Args:
        video_path: 视频文件路径 (未直接使用，由 hub 管理)。
        device: 推理设备 (未直接使用)。
        hub: FeatureHub 实例。

    Returns:
        逐帧 AU 强度字典列表，每个字典为 {AU名称: 强度值}。
        无法提取时返回空列表。
    """
    if hub is None:
        logger.warning("au_features 需要 hub 参数以获取视频帧缓存")
        return []

    frames = hub.get("video_frames")
    if not frames:
        return []

    try:
        from src.expression_naturalness.au_extractor import AUExtractor

        extractor = AUExtractor()
        return extractor.extract_sequence(frames)
    except (ImportError, Exception) as e:
        logger.warning("AU 特征提取失败 (%s)", e)
        return []
