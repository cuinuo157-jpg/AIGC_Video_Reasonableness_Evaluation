"""OpenAI 兼容本地 VLLM：抽帧与 chat.completions 调用（与 scripts/test_qwen_35_video.py 一致）。"""

from __future__ import annotations

import base64
import logging
from typing import Any

import cv2

logger = logging.getLogger(__name__)


def extract_frames_jpeg_bytes(video_path: str, fps: int = 2) -> list[bytes]:
    """按指定 fps 从视频抽帧，返回 JPEG 字节列表。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 25.0
    frame_interval = max(1, int(original_fps / fps))

    frames: list[bytes] = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                frames.append(buf.tobytes())
        frame_count += 1
    cap.release()
    logger.debug(
        "vllm extract_frames: %s frames from %s source frames (target_fps=%s)",
        len(frames),
        frame_count,
        fps,
    )
    return frames


def subsample_uniform(items: list[bytes], max_n: int) -> list[bytes]:
    if max_n < 1 or len(items) <= max_n:
        return items
    step = len(items) / max_n
    return [items[int(i * step)] for i in range(max_n)]


def frames_bytes_to_base64(frames: list[bytes]) -> list[str]:
    return [base64.b64encode(f).decode("utf-8") for f in frames]


def build_user_content_image_then_text(
    b64_jpegs: list[str], question: str
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for b64 in b64_jpegs:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )
    content.append({"type": "text", "text": question})
    return content


def chat_completions_text(
    *,
    base_url: str,
    api_key: str,
    model: str,
    user_content: list[dict[str, Any]],
    temperature: float,
    timeout: float,
) -> str:
    import openai

    client = openai.OpenAI(api_key=api_key, base_url=base_url.rstrip("/"))
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_content}],
        temperature=temperature,
        timeout=timeout,
    )
    msg = completion.choices[0].message
    return (msg.content or "").strip()
