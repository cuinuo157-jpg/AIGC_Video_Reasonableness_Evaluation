from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError, TypeError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError, TypeError):
        return default


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass
class MLLMConfig:
    """多模态大模型配置。

    各字段的默认值可通过同名环境变量覆盖（大写、加 MLLM_ 前缀），
    例如 MLLM_PROVIDER、MLLM_MODEL、MLLM_FPS 等。

    特殊别名（向后兼容）:
      DASHSCOPE_API_KEY  → api_key
      DASHSCOPE_BASE_URL → api_base_url
      VLLM_API_KEY       → api_key（次优先）
      VLLM_OPENAI_BASE_URL → api_base_url（次优先）
    """

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

    # ── 内部映射：字段名 → (env_优先, env_次优先, 类型转换) ──
    _ENV_MAP: dict[str, tuple[tuple[str, ...], type]] = {
        "backend": (("MLLM_BACKEND",), str),
        "local_model": (("MLLM_LOCAL_MODEL",), str),
        "local_model_path": (("MLLM_LOCAL_MODEL_PATH",), str),
        "api_provider": (("MLLM_PROVIDER",), str),
        "api_model": (("MLLM_MODEL",), str),
        "api_key": (("DASHSCOPE_API_KEY", "VLLM_API_KEY"), str),
        "api_base_url": (("DASHSCOPE_BASE_URL", "VLLM_OPENAI_BASE_URL"), str),
        "dashscope_video_fps": (("MLLM_FPS",), int),
        "max_frames": (("MLLM_MAX_FRAMES",), int),
        "vllm_max_frames": (("MLLM_VLLM_MAX_FRAMES",), int),
        "vllm_timeout": (("MLLM_TIMEOUT",), int),
        "temperature": (("MLLM_TEMPERATURE",), float),
        "device": (("MLLM_DEVICE",), str),
        "qwen_vl_device": (("MLLM_QWEN_VL_DEVICE",), str),
        "qwen_vl_model_path": (("MLLM_QWEN_VL_MODEL_PATH",), str),
    }

    def __post_init__(self) -> None:
        cls = type(self)
        for field_name, (env_names, type_fn) in self._ENV_MAP.items():
            current = getattr(self, field_name)
            default = getattr(cls, field_name, None)
            # 仅在字段值 == 类默认值时尝试 env 覆盖（即未被调用方显式传入）
            if current != default:
                continue
            for env_name in env_names:
                val = os.environ.get(env_name)
                if val is not None:
                    try:
                        setattr(self, field_name, type_fn(val.strip()))
                    except (ValueError, TypeError):
                        pass
                    break

    @classmethod
    def from_env(cls: type[MLLMConfig]) -> MLLMConfig:
        """从环境变量创建 MLLMConfig（忽略所有代码默认值，完全由 env 驱动）。"""
        return cls()  # __post_init__ 会处理 env 覆盖
