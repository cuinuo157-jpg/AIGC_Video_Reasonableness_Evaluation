from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .prompts import BIOLOGICAL_ANOMALY_PROMPT
from .roi_utils import extract_suspicious_rois


def _judge_rois_with_mllm(crop_frames: list[np.ndarray], mllm_client: Any) -> dict:
    provider = getattr(getattr(mllm_client, "config", None), "api_provider", "")
    if provider in ("dashscope", "vllm") and hasattr(mllm_client, "judge_video_path"):
        tmp_dir = tempfile.mkdtemp(prefix="bio_anomaly_dashscope_")
        video_path = Path(tmp_dir) / "roi_clip.mp4"
        try:
            h, w = crop_frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                6.0,
                (w, h),
            )
            if not writer.isOpened():
                raise RuntimeError("无法创建临时 ROI 视频文件")
            for frame in crop_frames:
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h))
                writer.write(frame)
            writer.release()
            return mllm_client.judge_video_path(str(video_path), BIOLOGICAL_ANOMALY_PROMPT)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return mllm_client.judge_video_clip(crop_frames, BIOLOGICAL_ANOMALY_PROMPT)


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
        result = _judge_rois_with_mllm(crop_frames, mllm_client)
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
