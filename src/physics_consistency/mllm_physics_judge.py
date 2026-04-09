from __future__ import annotations

from typing import Any

import numpy as np

from src.mllm.prompts.physics_commonsense import build_physics_prompt


def judge_physics_mllm(
    hub: Any,
    mllm_client: Any,
    drift_events: list[dict] | None = None,
) -> dict:
    """VLM 物理常识判定，支持漂移上下文注入和 provider 路由。"""
    prompt = build_physics_prompt(drift_events=drift_events)

    # DashScope 优先走视频路径接口
    provider = getattr(getattr(mllm_client, "config", None), "api_provider", "")
    if provider == "dashscope" and hasattr(mllm_client, "judge_video_path"):
        video_path = getattr(hub, "video_path", None)
        if video_path:
            try:
                return mllm_client.judge_video_path(video_path, prompt)
            except Exception:
                pass  # 降级到帧模式

    # 帧模式
    try:
        frames = hub.get("video_frames")
    except (KeyError, Exception):
        return {"skipped": True, "reason": "no video frames"}

    try:
        return mllm_client.judge_video_clip(frames, prompt)
    except Exception:
        return {"skipped": True, "reason": "mllm call failed"}
