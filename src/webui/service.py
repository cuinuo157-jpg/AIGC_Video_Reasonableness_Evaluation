from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.evaluation_pipeline import DEFAULT_ANOMALY_TYPES, DEFAULT_WEIGHTS, EvaluationPipeline
from src.feature_hub import VideoProcessingConfig

DIMENSION_CATALOG: dict[str, dict[str, str]] = {
    "face_identity": {
        "label": "身份一致性",
        "description": "检查主角外观与身份是否在时序上稳定。",
        "scope": "anomaly",
    },
    "expression": {
        "label": "表情自然度",
        "description": "评估面部动作单元组合和表情过渡是否自然。",
        "scope": "anomaly",
    },
    "biological_anomaly": {
        "label": "生物特征异常",
        "description": "检测眼、嘴、手、骨骼等人体结构异常。",
        "scope": "anomaly",
    },
    "motion_logic": {
        "label": "运动逻辑",
        "description": "评估动态幅度、平滑度和运动自然性。",
        "scope": "anomaly",
    },
    "physics": {
        "label": "物理常识",
        "description": "检查漂移、悬浮和违反物理规律的现象。",
        "scope": "anomaly",
    },
    "temporal_coherence": {
        "label": "时间一致性",
        "description": "检测物体异常出现、消失与跳变。",
        "scope": "full",
    },
    "background": {
        "label": "背景一致性",
        "description": "评估背景静态区域、深度和匹配稳定性。",
        "scope": "full",
    },
    "perceptual_quality": {
        "label": "感知质量",
        "description": "检查模糊、波动与画面瑕疵。",
        "scope": "full",
    },
}

FULL_DIMENSIONS = tuple(DEFAULT_WEIGHTS.keys())
DEFAULT_DEVICE = "cuda"
DEFAULT_SAMPLE_STRIDE = 2
DEFAULT_MAX_FRAMES = 48
DEFAULT_MAX_SIDE = 640
DEFAULT_UPLOAD_DIR = Path("outputs") / "webui_uploads"


@dataclass(frozen=True)
class WebUIRunConfig:
    video_path: str
    scope: str
    selected_dimensions: tuple[str, ...]
    device: str = DEFAULT_DEVICE
    enable_mllm: bool = False
    parallel: bool = True
    max_workers: int | None = None
    video_config: VideoProcessingConfig = field(
        default_factory=lambda: VideoProcessingConfig(
            sample_stride=DEFAULT_SAMPLE_STRIDE,
            max_frames=DEFAULT_MAX_FRAMES,
            max_side=DEFAULT_MAX_SIDE,
        )
    )


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "y", "on"}


def _coerce_int(
    value: Any,
    default: int | None,
    *,
    minimum: int | None = None,
    allow_none: bool = False,
) -> int | None:
    if value in (None, ""):
        return None if allow_none else default
    parsed = int(value)
    if minimum is not None:
        parsed = max(minimum, parsed)
    return parsed


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for item in value:
            items.extend(_coerce_list(item))
        return items
    text = str(value).strip()
    return [text] if text else []


def available_dimensions(scope: str) -> list[dict[str, str]]:
    keys = DEFAULT_ANOMALY_TYPES if scope == "anomaly" else FULL_DIMENSIONS
    return [{"key": key, **DIMENSION_CATALOG[key]} for key in keys]


def build_frontend_config() -> dict[str, Any]:
    return {
        "scopes": [
            {
                "key": "anomaly",
                "label": "五类异常",
                "description": "身份、表情、生物异常、运动逻辑、物理常识。",
                "dimensions": available_dimensions("anomaly"),
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
            "anomaly_types": list(DEFAULT_ANOMALY_TYPES),
            "selected_dimensions": list(FULL_DIMENSIONS),
        },
    }


def _normalize_dimensions(raw: list[str], scope: str) -> tuple[str, ...]:
    allowed = DEFAULT_ANOMALY_TYPES if scope == "anomaly" else FULL_DIMENSIONS
    selected = [item for item in raw if item in allowed]
    if not selected:
        selected = list(allowed)
    deduped: list[str] = []
    for item in selected:
        if item not in deduped:
            deduped.append(item)
    return tuple(deduped)


def build_run_config(payload: dict[str, Any], uploaded_video_path: str | None = None) -> WebUIRunConfig:
    video_path = uploaded_video_path or str(payload.get("video_path", "")).strip()
    if not video_path:
        raise ValueError("请提供视频文件或本地视频路径")
    if not Path(video_path).exists():
        raise ValueError(f"视频不存在: {video_path}")

    scope = "full" if str(payload.get("scope", "")).strip().lower() == "full" else "anomaly"
    selection_key = "selected_dimensions" if scope == "full" else "anomaly_types"
    selected_dimensions = _normalize_dimensions(_coerce_list(payload.get(selection_key)), scope)
    device = str(payload.get("device", DEFAULT_DEVICE)).strip() or DEFAULT_DEVICE

    video_config = VideoProcessingConfig(
        sample_stride=_coerce_int(payload.get("sample_stride"), DEFAULT_SAMPLE_STRIDE, minimum=1) or 1,
        max_frames=_coerce_int(payload.get("max_frames"), DEFAULT_MAX_FRAMES, minimum=2, allow_none=True),
        max_side=_coerce_int(payload.get("max_side"), DEFAULT_MAX_SIDE, minimum=64, allow_none=True),
    )

    max_workers = _coerce_int(payload.get("max_workers"), None, minimum=1, allow_none=True)

    return WebUIRunConfig(
        video_path=video_path,
        scope=scope,
        selected_dimensions=selected_dimensions,
        device=device,
        enable_mllm=_coerce_bool(payload.get("enable_mllm"), False),
        parallel=_coerce_bool(payload.get("parallel"), True),
        max_workers=max_workers,
        video_config=video_config,
    )


def run_analysis(config: WebUIRunConfig) -> tuple[Any, float]:
    pipeline = EvaluationPipeline(
        device=config.device,
        enable_mllm=config.enable_mllm,
        video_config=config.video_config,
        parallel=config.parallel,
        max_workers=config.max_workers,
    )
    start_time = time.perf_counter()
    if config.scope == "anomaly":
        report = pipeline.detect_anomalies(
            config.video_path,
            anomaly_types=config.selected_dimensions,
            parallel=config.parallel,
            max_workers=config.max_workers,
        )
    else:
        report = pipeline.evaluate(
            config.video_path,
            selected_dimensions=config.selected_dimensions,
            parallel=config.parallel,
            max_workers=config.max_workers,
        )
    return report, time.perf_counter() - start_time


def build_upload_path(filename: str, upload_dir: Path | None = None) -> Path:
    target_dir = (upload_dir or DEFAULT_UPLOAD_DIR).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name or "upload.mp4"
    safe_name = safe_name.replace(" ", "_")
    stem = Path(safe_name).stem[:80] or "video"
    suffix = Path(safe_name).suffix or ".mp4"
    return target_dir / f"{int(time.time())}_{stem}{suffix}"


def save_uploaded_file(file_obj: Any, filename: str, upload_dir: Path | None = None) -> str:
    target = build_upload_path(filename, upload_dir=upload_dir)
    with target.open("wb") as output:
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    return os.fspath(target)
