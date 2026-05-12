from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar


# ── 内部映射：字段名 → (env_优先, env_次优先, 类型转换) ──
# 模块级常量，供 from_env() 和 _apply_env 使用。
_ENV_MAP: dict[str, tuple[tuple[str, ...], type]] = {
    "backend": (("MLLM_BACKEND",), str),
    "local_model": (("MLLM_LOCAL_MODEL",), str),
    "local_model_path": (("MLLM_LOCAL_MODEL_PATH",), str),
    "api_provider": (("MLLM_PROVIDER",), str),
    "api_model": (("MLLM_MODEL",), str),
    "api_key": (("MLLM_API_KEY", "DASHSCOPE_API_KEY", "VLLM_API_KEY"), str),
    "api_base_url": (("MLLM_API_BASE_URL", "DASHSCOPE_BASE_URL", "VLLM_OPENAI_BASE_URL"), str),
    "api_service_name": (("MLLM_API_SERVICE_NAME",), str),
    "api_system_prompt": (("MLLM_API_SYSTEM_PROMPT",), str),
    "dashscope_video_fps": (("MLLM_FPS",), int),
    "max_frames": (("MLLM_MAX_FRAMES",), int),
    "vllm_max_frames": (("MLLM_VLLM_MAX_FRAMES",), int),
    "vllm_timeout": (("MLLM_TIMEOUT",), int),
    "temperature": (("MLLM_TEMPERATURE",), float),
    "device": (("MLLM_DEVICE",), str),
    "qwen_vl_device": (("MLLM_QWEN_VL_DEVICE",), str),
    "qwen_vl_model_path": (("MLLM_QWEN_VL_MODEL_PATH",), str),
}


def _read_env_or(field_name: str, default: str | int | float | None):
    """读取环境变量，若不存在则返回默认值。"""
    entry = _ENV_MAP.get(field_name)
    if entry is None:
        return default
    env_names, type_fn = entry
    for env_name in env_names:
        val = os.environ.get(env_name)
        if val is not None:
            try:
                return type_fn(val.strip())
            except (ValueError, TypeError):
                pass
    return default


@dataclass
class MLLMConfig:
    """多模态大模型配置。

    各字段的默认值可通过同名环境变量覆盖（大写、加 MLLM_ 前缀），
    例如 MLLM_PROVIDER、MLLM_MODEL、MLLM_FPS 等。

    特殊别名（向后兼容）:
      MLLM_API_KEY       → api_key
      MLLM_API_BASE_URL  → api_base_url
      DASHSCOPE_API_KEY  → api_key
      DASHSCOPE_BASE_URL → api_base_url
      VLLM_API_KEY       → api_key（次优先）
      VLLM_OPENAI_BASE_URL → api_base_url（次优先）

    使用建议:
      - 脚本中通过 CLI 参数 + 环境变量构建，见各脚本 build_mllm_client()
      - 也可直接用 MLLMConfig.from_env() 从 .env 构建
    """

    backend: str = "api"
    local_model: str = "InternVL2-8B"
    local_model_path: str | None = None
    api_provider: str = "vllm"
    api_model: str = "qwen3.5:9b"
    api_key: str | None = None
    api_base_url: str | None = None
    api_service_name: str = "aigc_video_reasonableness_evaluation"
    api_system_prompt: str = "你是一位专业的 AI 助手"
    dashscope_video_fps: int = 2
    max_frames: int = 16
    vllm_max_frames: int = 5
    vllm_timeout: int = 300
    temperature: float = 0.1
    device: str = "cuda"
    qwen_vl_model_path: str | None = None
    qwen_vl_device: str = "cuda"

    @classmethod
    def from_env(cls: type[MLLMConfig]) -> MLLMConfig:
        """完全从环境变量构建（忽略代码默认值）。"""
        kwargs = {}
        for field_name in _ENV_MAP:
            kwargs[field_name] = _read_env_or(field_name, getattr(cls, field_name, None))
        return cls(**kwargs)
