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
    try:
        frames = hub.get("video_frames")
    except KeyError:
        return {"skipped": True, "reason": "no video frames"}
    from src.mllm.prompts import MOTION_NATURALNESS_PROMPT

    return mllm_client.judge_video_clip(frames, MOTION_NATURALNESS_PROMPT)
