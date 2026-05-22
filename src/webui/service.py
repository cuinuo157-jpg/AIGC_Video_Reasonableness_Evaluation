"""WebUI service layer — thin adapter over src.api.core.

All core analysis, job management, and reporting logic lives in src.api.core.
This module provides webui-specific helpers:
  - frontend config builder
  - file upload helpers
  - backward-compatible type aliases (WebUIRunConfig, WebUIJob, WebUIJobManager)
  - build_run_config (wraps parse_analysis_config for form data)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from src.api.core import (
    DEFAULT_AU_BACKEND,
    DEFAULT_AU_EXTERNAL_PYTHON,
    DEFAULT_DEVICE,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MAX_SIDE,
    DEFAULT_SAMPLE_STRIDE,
    DEFAULT_TOP5_TYPES,
    DEFAULT_UPLOAD_DIR,
    DEFAULT_VISUALIZATION_DIR,
    DIMENSION_CATALOG,
    FULL_DIMENSIONS,
    SUPPORTED_MLLM_PROVIDERS,
    _default_mllm_config,
    _safe_stem,
    available_dimensions,
    parse_analysis_config,
    run_analysis,
    run_analysis_with_hub,
    scan_video_directory,
)
from src.api.core import AnalysisConfig as _AnalysisConfig
from src.api.core import Job as _Job
from src.api.core import JobManager as _JobManager
from src.evaluation_pipeline import DEFAULT_ANOMALY_TYPES
from src.mllm import MLLMConfig

# ── Backward-compatible type aliases ─────────────────────────────────
WebUIRunConfig = _AnalysisConfig
WebUIJob = _Job
WebUIJobManager = _JobManager

# ── Re-export constants (for external consumers) ─────────────────────
__all__ = [
    "DEFAULT_AU_BACKEND",
    "DEFAULT_AU_EXTERNAL_PYTHON",
    "DEFAULT_DEVICE",
    "DEFAULT_MAX_FRAMES",
    "DEFAULT_MAX_SIDE",
    "DEFAULT_RESULTS_DIR",
    "DEFAULT_SAMPLE_STRIDE",
    "DEFAULT_UPLOAD_DIR",
    "DEFAULT_VISUALIZATION_DIR",
    "DIMENSION_CATALOG",
    "FULL_DIMENSIONS",
    "SUPPORTED_MLLM_PROVIDERS",
    "WebUIJob",
    "WebUIJobManager",
    "WebUIRunConfig",
    "available_dimensions",
    "build_frontend_config",
    "build_run_config",
    "build_upload_path",
    "run_analysis",
    "run_analysis_with_hub",
    "save_uploaded_file",
    "scan_video_directory",
]

# Re-export the results dir
DEFAULT_RESULTS_DIR = Path("outputs") / "webui_results"


# ── WebUI-specific functions ─────────────────────────────────────────

def build_frontend_config() -> dict[str, Any]:
    """Build frontend configuration dict for the web UI."""
    mllm_defaults = _default_mllm_config()
    return {
        "scopes": [
            {
                "key": "anomaly",
                "label": "五类异常",
                "description": "身份、表情、生物异常、运动逻辑、物理常识。",
                "dimensions": available_dimensions("anomaly"),
            },
            {
                "key": "top5",
                "label": "Top5 场景",
                "description": "身份、生物异常、运动逻辑、物理常识、时间一致性。",
                "dimensions": available_dimensions("top5"),
            },
            {
                "key": "full",
                "label": "全量维度",
                "description": "包含时间一致性、背景一致性与感知质量。",
                "dimensions": available_dimensions("full"),
            },
        ],
        "defaults": {
            "device": DEFAULT_DEVICE,
            "parallel": True,
            "sample_stride": DEFAULT_SAMPLE_STRIDE,
            "max_frames": DEFAULT_MAX_FRAMES,
            "max_side": DEFAULT_MAX_SIDE,
            "au_backend": DEFAULT_AU_BACKEND,
            "au_external_python": DEFAULT_AU_EXTERNAL_PYTHON,
            "save_visualizations": False,
            "visualization_root": os.fspath(DEFAULT_VISUALIZATION_DIR),
            "enable_mllm": False,
            "mllm_provider": "huawei_custom",
            "mllm_model": "Qwen3-VL-235B-A22B-Instruct",
            "mllm_base_url": "http://aitest-beta.rnd.huawei.com/v1",
            "mllm_api_key": "",
            "mllm_service_name": "simple_client",
            "anomaly_types": list(DEFAULT_ANOMALY_TYPES),
            "top5_types": list(DEFAULT_TOP5_TYPES),
            "selected_dimensions": list(FULL_DIMENSIONS),
        },
        "mllm_providers": list(SUPPORTED_MLLM_PROVIDERS),
        "huawei_models": [
            "Qwen2.5-72B",
            "Qwen2.5-VL-32B-Instruct",
            "Qwen2.5-VL-72B-Instruct",
            "DeepSeek-V3",
            "Qwen3-235B-A22B-Instruct-2507",
            "Qwen3-VL-32B-Instruct",
            "Qwen3-VL-235B-A22B-Instruct",
        ],
    }


def build_run_config(
    payload: dict[str, Any],
    uploaded_video_path: str | None = None,
) -> WebUIRunConfig:
    """Build a WebUIRunConfig (AnalysisConfig) from form data payload."""
    return parse_analysis_config(payload, uploaded_video_path=uploaded_video_path)


def build_upload_path(
    filename: str, upload_dir: Path | None = None
) -> Path:
    target_dir = (upload_dir or DEFAULT_UPLOAD_DIR).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name or "upload.mp4"
    safe_name = safe_name.replace(" ", "_")
    stem = _safe_stem(safe_name, max_len=80)
    suffix = Path(safe_name).suffix or ".mp4"
    return target_dir / f"{int(time.time())}_{stem}{suffix}"


def save_uploaded_file(
    file_obj: Any, filename: str, upload_dir: Path | None = None
) -> str:
    target = build_upload_path(filename, upload_dir=upload_dir)
    with target.open("wb") as output:
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    return os.fspath(target)
