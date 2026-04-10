from __future__ import annotations

from typing import Any

import numpy as np


def judge_naturalness_mllm(
    hub: Any,
    mllm_client: Any,
    flows: list[tuple[np.ndarray, np.ndarray]],
    smoothness_score: float,
) -> dict:
    if smoothness_score > 0.8:
        return {"skipped": True, "reason": "smoothness above threshold"}
    from src.mllm.prompts import MOTION_NATURALNESS_PROMPT

    provider = getattr(getattr(mllm_client, "config", None), "api_provider", "")
    if provider in ("dashscope", "vllm") and hasattr(mllm_client, "judge_video_path"):
        video_path = getattr(hub, "video_path", None)
        if video_path:
            return mllm_client.judge_video_path(video_path, MOTION_NATURALNESS_PROMPT)

    try:
        frames = hub.get("video_frames")
    except KeyError:
        return {"skipped": True, "reason": "no video frames"}
    return mllm_client.judge_video_clip(frames, MOTION_NATURALNESS_PROMPT)
