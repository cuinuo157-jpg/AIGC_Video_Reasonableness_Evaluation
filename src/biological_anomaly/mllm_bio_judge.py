from __future__ import annotations

from typing import Any

import numpy as np

from .prompts import BIOLOGICAL_ANOMALY_PROMPT
from .roi_utils import extract_suspicious_rois


def judge_biological_anomaly_mllm(
    frames: list[np.ndarray],
    keypoints_seq: list[dict],
    suspicious_frames: list[dict],
    mllm_client: Any,
    max_crops: int = 8,
) -> dict:
    """Level 3: 使用 MLLM 对疑似异常区域进行语义判定。

    Args:
        frames: 视频全部帧（BGR）
        keypoints_seq: 关键点序列
        suspicious_frames: 疑似异常帧信息列表
        mllm_client: MLLMClient 实例
        max_crops: 最大 ROI 裁剪数

    Returns:
        {"skipped": bool, "has_anomalies": bool, "anomalies": list[dict]}
    """
    if not suspicious_frames or mllm_client is None:
        return {"skipped": True, "has_anomalies": False, "anomalies": []}

    # 提取 ROI 裁剪
    rois = extract_suspicious_rois(
        frames, keypoints_seq, suspicious_frames, max_crops=max_crops
    )

    if not rois:
        return {"skipped": True, "has_anomalies": False, "anomalies": []}

    # 收集裁剪帧送入 MLLM
    crop_frames = [roi["crop"] for roi in rois]

    try:
        result = mllm_client.judge_video_clip(crop_frames, BIOLOGICAL_ANOMALY_PROMPT)
    except Exception:
        return {"skipped": True, "has_anomalies": False, "anomalies": []}

    if result.get("skipped"):
        return {"skipped": True, "has_anomalies": False, "anomalies": []}

    anomalies = result.get("anomalies", [])
    # 为每个异常补充来源帧信息
    for i, anomaly in enumerate(anomalies):
        if i < len(rois):
            anomaly["source_frame_idx"] = rois[i]["frame_idx"]
            anomaly["source_region"] = rois[i]["region_type"]

    return {
        "skipped": False,
        "has_anomalies": result.get("has_anomalies", bool(anomalies)),
        "anomalies": anomalies,
    }
