#!/usr/bin/env python3
"""本地 VLLM 视频抽帧推理自测脚本（核心逻辑见 src.mllm.vllm_openai_video）。

配置方式（优先级从高到低）:
    1. 调用参数显式传入
    2. .env 环境变量（通过 MLLMConfig 读取）
    3. MLLMConfig 硬编码默认值
"""

import logging
from datetime import datetime
from pathlib import Path

from src.mllm.config import MLLMConfig
from src.mllm.dotenv_loader import load_dotenv
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
    cfg: MLLMConfig,
    *,
    save_frames: bool = True,
    frames_output_dir: str = "/data/AIGC_Video_Reasonableness_Evaluation/tests/video_frames",
) -> str | None:
    """使用 MLLMConfig 配置对视频抽帧并调用 VLLM 推理。

    Args:
        video_path: 视频文件路径。
        question: 推理 prompt。
        cfg: MLLM 配置（从 .env 加载，可被调用方覆盖）。
        save_frames: 是否保存抽帧结果到磁盘。
        frames_output_dir: 抽帧保存根目录。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        frames = extract_frames_jpeg_bytes(video_path, cfg.dashscope_video_fps)
        if not frames:
            logger.error("未抽到任何帧")
            return None
        frames = subsample_uniform(frames, cfg.vllm_max_frames)
        if save_frames:
            save_frames_debug(frames, frames_output_dir, video_path, timestamp)
        b64_frames = frames_bytes_to_base64(frames)
    except Exception as e:
        logger.error("预处理失败: %s", e)
        return None

    # MLLMConfig.api_base_url 级联读取 MLLM_API_BASE_URL → DASHSCOPE_BASE_URL → VLLM_OPENAI_BASE_URL
    base_url = (cfg.api_base_url or "http://localhost:8201/v1").rstrip("/")
    api_key = cfg.api_key or "not-needed"
    content = build_user_content_image_then_text(b64_frames, question)
    try:
        logger.info("发起推理请求 (模型: %s, 帧数: %s)", cfg.api_model, len(b64_frames))
        return chat_completions_text(
            base_url=base_url,
            api_key=api_key,
            model=cfg.api_model,
            user_content=content,
            temperature=cfg.temperature,
            timeout=float(cfg.vllm_timeout),
        )
    except Exception as e:
        logger.error("推理失败: %s", e)
        return None


if __name__ == "__main__":
    load_dotenv()
    cfg = MLLMConfig.from_env()

    video_path = "/data/AIGC_Video_Reasonableness_Evaluation/data/videos/The camera orbits around. Airpods Max, the camera circles around.-0.mp4"
    question = """请分析AI生成的视频内容，若发现异常请明确指出。"""
    result = analyze_video_by_frames(
        video_path=video_path,
        question=question,
        cfg=cfg,
        save_frames=True,
        frames_output_dir="/data/AIGC_Video_Reasonableness_Evaluation/tests/video_frames",
    )
    if result:
        print("\n=== 模型返回 ===")
        print(result)
    else:
        print("分析失败，请检查日志")
