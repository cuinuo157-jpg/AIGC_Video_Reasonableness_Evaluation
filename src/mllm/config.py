from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MLLMConfig:
    backend: str = "api"  # "local", "api", "hybrid"
    local_model: str = "InternVL2-8B"  # "InternVL2-8B", "Qwen-VL"
    local_model_path: str | None = None
    api_provider: str = "openai"  # "openai", "anthropic"
    api_model: str = "gpt-4o"
    api_key: str | None = None
    api_base_url: str | None = None
    max_frames: int = 16
    temperature: float = 0.1
    device: str = "cuda"
    # Qwen-VL 本地模型配置（预留）
    qwen_vl_model_path: str | None = None
    qwen_vl_device: str = "cuda"
