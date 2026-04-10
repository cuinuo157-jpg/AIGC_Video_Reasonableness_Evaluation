from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MLLMConfig:
    backend: str = "api"  # "local", "api", "hybrid"
    local_model: str = "InternVL2-8B"  # "InternVL2-8B", "Qwen-VL"
    local_model_path: str | None = None
    api_provider: str = "vllm"  # "vllm", "openai", "anthropic", "dashscope"
    api_model: str = "qwen3.5:9b"
    api_key: str | None = None
    api_base_url: str | None = None
    dashscope_video_fps: int = 2
    max_frames: int = 16
    vllm_max_frames: int = 5
    vllm_timeout: int = 300
    temperature: float = 0.1
    device: str = "cuda"
    # Qwen-VL 本地模型配置（预留）
    qwen_vl_model_path: str | None = None
    qwen_vl_device: str = "cuda"
