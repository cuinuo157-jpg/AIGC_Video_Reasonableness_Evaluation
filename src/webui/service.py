from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.evaluation_pipeline import DEFAULT_ANOMALY_TYPES, DEFAULT_WEIGHTS, EvaluationPipeline
from src.feature_hub import VideoProcessingConfig

logger = logging.getLogger(__name__)

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
DEFAULT_RESULTS_DIR = Path("outputs") / "webui_results"
MAX_LOG_LINES = 800


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


@dataclass
class WebUIJob:
    job_id: str
    run_config: WebUIRunConfig
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    status: str = "queued"
    logs: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    result_json_path: str | None = None
    log_path: str | None = None
    error: str | None = None
    completed_at: float | None = None


class _JobLogWriter:
    def __init__(self, append_fn: Any) -> None:
        self._append = append_fn
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.rstrip():
                self._append(line.rstrip())
        return len(text)

    def flush(self) -> None:
        if self._buffer.rstrip():
            self._append(self._buffer.rstrip())
        self._buffer = ""


class _TeeWriter:
    def __init__(self, *writers: Any) -> None:
        self._writers = writers

    def write(self, text: str) -> int:
        for writer in self._writers:
            writer.write(text)
        return len(text)

    def flush(self) -> None:
        for writer in self._writers:
            flush_fn = getattr(writer, "flush", None)
            if callable(flush_fn):
                flush_fn()


class _JobLogHandler(logging.Handler):
    def __init__(self, append_fn: Any) -> None:
        super().__init__()
        self._append = append_fn
        self.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._append(self.format(record))
        except Exception:
            self.handleError(record)


class WebUIJobManager:
    def __init__(self, results_dir: Path | None = None) -> None:
        self.results_dir = (results_dir or DEFAULT_RESULTS_DIR).resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, WebUIJob] = {}
        self._lock = threading.RLock()

    def create_job(self, run_config: WebUIRunConfig) -> WebUIJob:
        job_id = uuid.uuid4().hex[:12]
        job = WebUIJob(job_id=job_id, run_config=run_config)
        with self._lock:
            self._jobs[job_id] = job
        self._append_log(job_id, f"[job] 已创建任务 {job_id}")
        worker = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        worker.start()
        return job

    def get_job(self, job_id: str) -> WebUIJob:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    def get_job_snapshot(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        with self._lock:
            return {
                "job_id": job.job_id,
                "status": job.status,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "completed_at": job.completed_at,
                "error": job.error,
                "result_json_path": job.result_json_path,
                "log_path": job.log_path,
                "has_result": job.result is not None,
                "result": job.result,
            }

    def get_job_logs(self, job_id: str, offset: int = 0) -> dict[str, Any]:
        job = self.get_job(job_id)
        with self._lock:
            safe_offset = max(0, min(offset, len(job.logs)))
            lines = job.logs[safe_offset:]
            return {
                "job_id": job.job_id,
                "offset": safe_offset,
                "next_offset": len(job.logs),
                "lines": lines,
                "completed": job.status in {"completed", "failed"},
            }

    def _append_log(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.logs.append(message)
            if len(job.logs) > MAX_LOG_LINES:
                overflow = len(job.logs) - MAX_LOG_LINES
                del job.logs[:overflow]
            job.updated_at = time.time()

    def _build_output_paths(self, job: WebUIJob) -> tuple[Path, Path]:
        stem = Path(job.run_config.video_path).stem[:80] or "video"
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(job.created_at))
        result_path = self.results_dir / f"{timestamp}_{stem}_{job.job_id}_report.json"
        log_path = self.results_dir / f"{timestamp}_{stem}_{job.job_id}.log"
        return result_path, log_path

    def _persist_job_outputs(
        self,
        job: WebUIJob,
        result_path: Path | None = None,
        log_path: Path | None = None,
    ) -> tuple[str, str]:
        if result_path is None or log_path is None:
            built_result_path, built_log_path = self._build_output_paths(job)
            result_path = result_path or built_result_path
            log_path = log_path or built_log_path
        if job.result is not None:
            result_path.write_text(
                json.dumps(job.result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        log_path.write_text("\n".join(job.logs), encoding="utf-8")
        return os.fspath(result_path), os.fspath(log_path)

    def _run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        with self._lock:
            job.status = "running"
            job.updated_at = time.time()

        def append(message: str) -> None:
            self._append_log(job_id, message)

        append(f"[job] 开始分析 {job.run_config.video_path}")
        append(
            f"[job] 范围={job.run_config.scope}, 维度={','.join(job.run_config.selected_dimensions)}, "
            f"device={job.run_config.device}, parallel={job.run_config.parallel}"
        )

        writer = _JobLogWriter(append)
        tee_stdout = _TeeWriter(sys.stdout, writer)
        tee_stderr = _TeeWriter(sys.stderr, writer)
        root_logger = logging.getLogger()
        log_handler = _JobLogHandler(append)
        root_logger.addHandler(log_handler)

        try:
            with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
                report, elapsed = run_analysis(job.run_config)
            from .reporting import build_dashboard_report

            writer.flush()
            result_path, log_path = self._build_output_paths(job)
            result_json_path = os.fspath(result_path)
            log_path_str = os.fspath(log_path)
            payload = build_dashboard_report(report, job.run_config, elapsed)
            payload["result_json_path"] = result_json_path
            payload["log_path"] = log_path_str
            job.result_json_path = result_json_path
            job.log_path = log_path_str
            job.result = payload
            append(f"[job] 结果已写入 {result_json_path}")
            append(f"[job] 日志已写入 {log_path_str}")
            self._persist_job_outputs(job, result_path, log_path)
            with self._lock:
                job.status = "completed"
                job.completed_at = time.time()
                job.updated_at = job.completed_at
        except Exception as exc:
            writer.flush()
            append(f"[job] 任务失败: {exc}")
            with self._lock:
                job.status = "failed"
                job.error = str(exc)
                job.completed_at = time.time()
                job.updated_at = job.completed_at
            try:
                result_path, log_path = self._build_output_paths(job)
                job.log_path = os.fspath(log_path)
                if job.result is not None:
                    job.result_json_path = os.fspath(result_path)
                self._persist_job_outputs(job, result_path, log_path)
            except Exception:
                logger.exception("persist failed job log")
            logger.exception("webui job failed: %s", job_id)
        finally:
            root_logger.removeHandler(log_handler)


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
