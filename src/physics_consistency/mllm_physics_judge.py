from __future__ import annotations

from typing import Any


def judge_physics_mllm(hub: Any, mllm_client: Any) -> dict:
    try:
        frames = hub.get("video_frames")
    except KeyError:
        return {"skipped": True}
    from src.mllm.prompts import PHYSICS_COMMONSENSE_PROMPT

    return mllm_client.judge_video_clip(frames, PHYSICS_COMMONSENSE_PROMPT)
