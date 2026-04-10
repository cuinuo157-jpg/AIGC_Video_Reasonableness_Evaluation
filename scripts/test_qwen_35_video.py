#!/usr/bin/env python3
"""本地 VLLM 视频抽帧推理自测脚本（核心逻辑见 src.mllm.vllm_openai_video）。"""

import logging
import os
from datetime import datetime
from pathlib import Path

from src.mllm.vllm_openai_video import (
    chat_completions_text,
    extract_frames_jpeg_bytes,
    frames_bytes_to_base64,
    build_user_content_image_then_text,
    subsample_uniform,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def save_frames_debug(
    frames: list[bytes], output_base_dir: str, video_path: str, timestamp: str
) -> str:
    video_name = Path(video_path).stem
    save_dir = Path(output_base_dir) / f"{timestamp}_{video_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    for i, frame_bytes in enumerate(frames):
        save_path = save_dir / f"frame_{i:03d}.jpg"
        with open(save_path, "wb") as f:
            f.write(frame_bytes)
    logger.info("关键帧已保存: %s (%s 帧)", save_dir, len(frames))
    return str(save_dir)


def analyze_video_by_frames(
    video_path: str,
    question: str,
    fps: int = 2,
    model: str = "qwen3.5:9b",
    max_frames: int = 5,
    save_frames: bool = True,
    frames_output_dir: str = "/data/AIGC_Video_Reasonableness_Evaluation/tests/video_frames",
) -> str | None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        frames = extract_frames_jpeg_bytes(video_path, fps)
        if not frames:
            logger.error("未抽到任何帧")
            return None
        frames = subsample_uniform(frames, max_frames)
        if save_frames:
            save_frames_debug(frames, frames_output_dir, video_path, timestamp)
        b64_frames = frames_bytes_to_base64(frames)
    except Exception as e:
        logger.error("预处理失败: %s", e)
        return None

    base_url = os.environ.get("VLLM_OPENAI_BASE_URL", "http://localhost:8201/v1")
    api_key = os.environ.get("VLLM_API_KEY", "not-needed")
    content = build_user_content_image_then_text(b64_frames, question)
    try:
        logger.info("发起推理请求 (模型: %s, 帧数: %s)", model, len(b64_frames))
        return chat_completions_text(
            base_url=base_url,
            api_key=api_key,
            model=model,
            user_content=content,
            temperature=0.1,
            timeout=300.0,
        )
    except Exception as e:
        logger.error("推理失败: %s", e)
        return None


if __name__ == "__main__":
    video_path = "/data/AIGC_Video_Reasonableness_Evaluation/data/videos/The camera orbits around. Airpods Max, the camera circles around.-0.mp4"
    question = """请分析AI生成的视频内容，若发现异常请明确指出。"""
    result = analyze_video_by_frames(
        video_path=video_path,
        question=question,
        fps=3,
        max_frames=5,
        save_frames=True,
        frames_output_dir="/data/AIGC_Video_Reasonableness_Evaluation/tests/video_frames",
    )
    if result:
        print("\n=== 模型返回 ===")
        print(result)
    else:
        print("分析失败，请检查日志")
