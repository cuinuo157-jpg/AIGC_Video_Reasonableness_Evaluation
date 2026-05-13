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
from src.expression_naturalness.au_extractor import (
    _DEFAULT_AU_BACKEND as DEFAULT_AU_BACKEND,
    _DEFAULT_AU_PYTHON as DEFAULT_AU_EXTERNAL_PYTHON,
)
from src.feature_hub import VideoProcessingConfig
from src.mllm import MLLMClient, MLLMConfig

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
DEFAULT_VISUALIZATION_DIR = Path("outputs") / "webui_artifacts"
MAX_LOG_LINES = 800
SUPPORTED_MLLM_PROVIDERS = ("vllm", "dashscope", "openai", "anthropic", "huawei_custom")
BATCH_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".MP4", ".AVI", ".MOV", ".MKV", ".WEBM")


def _default_mllm_config() -> MLLMConfig:
    return MLLMConfig.from_env()


def scan_video_directory(
    dir_path: str,
    extensions: tuple[str, ...] | None = None,
    recursive: bool = False,
) -> list[str]:
    """Scan a directory for video files. Returns sorted list of absolute paths."""
    exts = extensions or BATCH_VIDEO_EXTENSIONS
    target = Path(dir_path).resolve()
    if not target.is_dir():
        raise ValueError(f"不是有效目录: {dir_path}")
    pattern = "**/*" if recursive else "*"
    videos: list[str] = []
    for ext in exts:
        for p in target.glob(pattern):
            if p.is_file() and p.suffix == ext:
                videos.append(str(p))
    return sorted(set(videos))


@dataclass(frozen=True)
class WebUIRunConfig:
    video_path: str
    scope: str
    selected_dimensions: tuple[str, ...]
    device: str = DEFAULT_DEVICE
    au_backend: str = DEFAULT_AU_BACKEND
    au_external_python: str = DEFAULT_AU_EXTERNAL_PYTHON
    enable_mllm: bool = False
    mllm_provider: str = MLLMConfig.api_provider
    mllm_model: str = MLLMConfig.api_model
    mllm_base_url: str | None = None
    mllm_api_key: str | None = None
    mllm_service_name: str = MLLMConfig.api_service_name
    save_visualizations: bool = False
    visualization_root: str | None = os.fspath(DEFAULT_VISUALIZATION_DIR)
    parallel: bool = True
    max_workers: int | None = None
    video_config: VideoProcessingConfig = field(
        default_factory=lambda: VideoProcessingConfig(
            sample_stride=DEFAULT_SAMPLE_STRIDE,
            max_frames=DEFAULT_MAX_FRAMES,
            max_side=DEFAULT_MAX_SIDE,
        )
    )
    video_dir: str | None = None
    file_extensions: str = ".mp4,.avi,.mov,.mkv,.webm"
    recursive_scan: bool = False


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
    artifact_root: str | None = None
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
                "artifact_root": job.artifact_root,
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

    def _build_artifact_dir(self, job: WebUIJob) -> Path:
        stem = Path(job.run_config.video_path).stem[:80] or "video"
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(job.created_at))
        base_dir = Path(job.run_config.visualization_root or DEFAULT_VISUALIZATION_DIR).resolve()
        return base_dir / f"{timestamp}_{stem}_{job.job_id}"

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

    def _run_batch_job(self, job: WebUIJob, video_list: list[str], append_fn: Any) -> None:
        """Run batch analysis sequentially for all videos in the list."""
        total = len(video_list)
        batch_results: list[dict[str, Any]] = []
        batch_start = time.perf_counter()

        for idx, video_path in enumerate(video_list, 1):
            video_name = Path(video_path).name
            append_fn(f"[batch] 处理 {idx}/{total}: {video_name}")

            single_config = WebUIRunConfig(
                video_path=video_path,
                scope=job.run_config.scope,
                selected_dimensions=job.run_config.selected_dimensions,
                device=job.run_config.device,
                au_backend=job.run_config.au_backend,
                au_external_python=job.run_config.au_external_python,
                enable_mllm=job.run_config.enable_mllm,
                mllm_provider=job.run_config.mllm_provider,
                mllm_model=job.run_config.mllm_model,
                mllm_base_url=job.run_config.mllm_base_url,
                mllm_api_key=job.run_config.mllm_api_key,
                mllm_service_name=job.run_config.mllm_service_name,
                save_visualizations=False,
                parallel=job.run_config.parallel,
                max_workers=job.run_config.max_workers,
                video_config=job.run_config.video_config,
            )

            try:
                report, elapsed = run_analysis(single_config)
                from .reporting import build_dashboard_report

                result_data = build_dashboard_report(report, single_config, elapsed)
                batch_results.append({
                    "video_name": video_name,
                    "video_path": video_path,
                    "final_score": result_data.get("final_score"),
                    "status": "completed",
                    "elapsed_sec": round(elapsed, 3),
                    "active_dimensions": result_data.get("active_dimensions", []),
                    "error": None,
                })
                append_fn(
                    f"[batch] ✓ {idx}/{total}: {video_name} - "
                    f"综合分 {result_data.get('final_score', 0):.3f} ({elapsed:.1f}s)"
                )
            except Exception as exc:
                batch_results.append({
                    "video_name": video_name,
                    "video_path": video_path,
                    "final_score": None,
                    "status": "failed",
                    "elapsed_sec": 0,
                    "active_dimensions": [],
                    "error": str(exc),
                })
                append_fn(f"[batch] ✗ {idx}/{total}: {video_name} - 失败: {exc}")

            # Update progress in job for frontend polling
            with self._lock:
                job.result = {
                    "batch": True,
                    "batch_progress": {
                        "current": idx,
                        "total": total,
                        "current_video": video_name,
                    },
                    "video_dir": job.run_config.video_dir,
                    "total_videos": total,
                    "completed_videos": len([r for r in batch_results if r["status"] == "completed"]),
                    "failed_videos": len([r for r in batch_results if r["status"] == "failed"]),
                    "video_results": list(batch_results),
                }

        total_elapsed = time.perf_counter() - batch_start

        from .reporting import build_batch_report

        final_report = build_batch_report(batch_results, job.run_config, total_elapsed)

        result_path, log_path = self._build_output_paths(job)
        job.result = final_report
        job.result_json_path = os.fspath(result_path)
        job.log_path = os.fspath(log_path)

        result_path.write_text(
            json.dumps(final_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log_path.write_text("\n".join(job.logs), encoding="utf-8")

        completed = final_report["completed_videos"]
        failed = final_report["failed_videos"]
        avg = final_report["aggregate"]["avg_score"]
        append_fn(
            f"[batch] 批量处理完成: {total} 个视频, "
            f"成功 {completed}, 失败 {failed}, "
            f"平均分 {avg:.3f}, 总耗时 {total_elapsed:.1f}s"
        )

        with self._lock:
            job.status = "completed"
            job.completed_at = time.time()
            job.updated_at = job.completed_at

    def _run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        with self._lock:
            job.status = "running"
            job.updated_at = time.time()

        def append(message: str) -> None:
            self._append_log(job_id, message)

        append(f"[job] 开始分析 {job.run_config.video_path or job.run_config.video_dir}")
        append(
            f"[job] 范围={job.run_config.scope}, 维度={','.join(job.run_config.selected_dimensions)}, "
            f"device={job.run_config.device}, parallel={job.run_config.parallel}"
        )
        append(
            f"[job] AU 路由: backend={job.run_config.au_backend}, "
            f"external_python={job.run_config.au_external_python or '-'}"
        )
        append(
            f"[job] 可视化产物: {'开启' if job.run_config.save_visualizations else '关闭'}, "
            f"root={job.run_config.visualization_root or '-'}"
        )
        if job.run_config.enable_mllm:
            append(
                f"[job] MLLM: provider={job.run_config.mllm_provider}, "
                f"model={job.run_config.mllm_model}, "
                f"base_url={job.run_config.mllm_base_url or '-'}, "
                f"service_name={job.run_config.mllm_service_name or '-'}"
            )

        if job.run_config.video_dir:
            extensions = tuple(
                e.strip() for e in job.run_config.file_extensions.split(",") if e.strip()
            ) or BATCH_VIDEO_EXTENSIONS
            video_list = scan_video_directory(
                job.run_config.video_dir,
                extensions=extensions,
                recursive=job.run_config.recursive_scan,
            )
            if not video_list:
                raise ValueError(f"目录中未找到匹配的视频文件: {job.run_config.video_dir}")
            append(f"[batch] 扫描到 {len(video_list)} 个视频文件")
            self._run_batch_job(job, video_list, append)
            return

        writer = _JobLogWriter(append)
        tee_stdout = _TeeWriter(sys.stdout, writer)
        tee_stderr = _TeeWriter(sys.stderr, writer)
        root_logger = logging.getLogger()
        log_handler = _JobLogHandler(append)
        root_logger.addHandler(log_handler)

        try:
            with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
                if job.run_config.save_visualizations:
                    artifact_dir = self._build_artifact_dir(job)
                    job.artifact_root = os.fspath(artifact_dir)
                    report, elapsed, hub = run_analysis_with_hub(job.run_config)
                else:
                    report, elapsed = run_analysis(job.run_config)
                    artifact_dir = None
                    hub = None
            from .reporting import build_dashboard_report

            writer.flush()
            result_path, log_path = self._build_output_paths(job)
            result_json_path = os.fspath(result_path)
            log_path_str = os.fspath(log_path)
            payload = build_dashboard_report(report, job.run_config, elapsed)
            if job.run_config.save_visualizations and artifact_dir is not None and hub is not None:
                from .artifacts import generate_visual_artifacts

                artifact_root, artifacts, artifacts_by_dimension = generate_visual_artifacts(
                    report,
                    job.run_config,
                    hub,
                    artifact_dir,
                    log_fn=append,
                )
                payload["artifact_root"] = artifact_root
                payload["artifacts"] = artifacts
                for card in payload["dimensions"]:
                    card["artifacts"] = artifacts_by_dimension.get(card["key"], [])
                job.artifact_root = artifact_root
                append(f"[job] 可视化产物已写入 {artifact_root}")
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
            "mllm_provider": mllm_defaults.api_provider,
            "mllm_model": mllm_defaults.api_model,
            "mllm_base_url": mllm_defaults.api_base_url or "",
            "mllm_api_key": "",
            "mllm_service_name": mllm_defaults.api_service_name,
            "anomaly_types": list(DEFAULT_ANOMALY_TYPES),
            "selected_dimensions": list(FULL_DIMENSIONS),
        },
        "mllm_providers": list(SUPPORTED_MLLM_PROVIDERS),
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
    mllm_defaults = _default_mllm_config()
    video_path = uploaded_video_path or str(payload.get("video_path", "")).strip()
    video_dir = str(payload.get("video_dir", "")).strip() or None
    if not video_path and not video_dir:
        raise ValueError("请提供视频文件、本地视频路径或视频目录")
    if video_path and not Path(video_path).exists():
        raise ValueError(f"视频不存在: {video_path}")
    if video_dir and not Path(video_dir).is_dir():
        raise ValueError(f"视频目录不存在: {video_dir}")

    scope = "full" if str(payload.get("scope", "")).strip().lower() == "full" else "anomaly"
    selection_key = "selected_dimensions" if scope == "full" else "anomaly_types"
    selected_dimensions = _normalize_dimensions(_coerce_list(payload.get(selection_key)), scope)
    device = str(payload.get("device", DEFAULT_DEVICE)).strip() or DEFAULT_DEVICE
    au_backend = str(payload.get("au_backend", DEFAULT_AU_BACKEND)).strip().lower() or DEFAULT_AU_BACKEND
    if au_backend not in {"local", "subprocess"}:
        raise ValueError(f"不支持的 AU 后端: {au_backend}")
    au_external_python = (
        str(payload.get("au_external_python", DEFAULT_AU_EXTERNAL_PYTHON)).strip()
        or DEFAULT_AU_EXTERNAL_PYTHON
    )
    enable_mllm = _coerce_bool(payload.get("enable_mllm"), False)
    mllm_provider = (
        str(payload.get("mllm_provider", mllm_defaults.api_provider)).strip().lower()
        or mllm_defaults.api_provider
    )
    if mllm_provider not in SUPPORTED_MLLM_PROVIDERS:
        raise ValueError(f"不支持的 MLLM provider: {mllm_provider}")
    mllm_model = (
        str(payload.get("mllm_model", mllm_defaults.api_model)).strip()
        or mllm_defaults.api_model
    )
    mllm_base_url = str(payload.get("mllm_base_url", mllm_defaults.api_base_url or "")).strip() or None
    mllm_api_key = (
        str(payload.get("mllm_api_key", mllm_defaults.api_key or "")).strip()
        or mllm_defaults.api_key
        or None
    )
    mllm_service_name = (
        str(payload.get("mllm_service_name", mllm_defaults.api_service_name)).strip()
        or mllm_defaults.api_service_name
    )
    save_visualizations = _coerce_bool(payload.get("save_visualizations"), False)
    visualization_root = (
        str(payload.get("visualization_root", os.fspath(DEFAULT_VISUALIZATION_DIR))).strip()
        or os.fspath(DEFAULT_VISUALIZATION_DIR)
    )

    video_config = VideoProcessingConfig(
        sample_stride=_coerce_int(payload.get("sample_stride"), DEFAULT_SAMPLE_STRIDE, minimum=1) or 1,
        max_frames=_coerce_int(payload.get("max_frames"), DEFAULT_MAX_FRAMES, minimum=2, allow_none=True),
        max_side=_coerce_int(payload.get("max_side"), DEFAULT_MAX_SIDE, minimum=64, allow_none=True),
    )
    max_workers = _coerce_int(payload.get("max_workers"), None, minimum=1, allow_none=True)
    file_extensions = str(payload.get("file_extensions", ".mp4,.avi,.mov,.mkv,.webm")).strip() or ".mp4,.avi,.mov,.mkv,.webm"
    recursive_scan = _coerce_bool(payload.get("recursive_scan"), False)

    return WebUIRunConfig(
        video_path=video_path,
        scope=scope,
        selected_dimensions=selected_dimensions,
        device=device,
        au_backend=au_backend,
        au_external_python=au_external_python,
        enable_mllm=enable_mllm,
        mllm_provider=mllm_provider,
        mllm_model=mllm_model,
        mllm_base_url=mllm_base_url,
        mllm_api_key=mllm_api_key,
        mllm_service_name=mllm_service_name,
        save_visualizations=save_visualizations,
        visualization_root=visualization_root,
        parallel=_coerce_bool(payload.get("parallel"), True),
        max_workers=max_workers,
        video_config=video_config,
        video_dir=video_dir,
        file_extensions=file_extensions,
        recursive_scan=recursive_scan,
    )


def run_analysis(config: WebUIRunConfig) -> tuple[Any, float]:
    mllm_client = None
    if config.enable_mllm:
        mllm_client = MLLMClient(
            MLLMConfig.from_env_with_overrides(
                backend="api",
                api_provider=config.mllm_provider,
                api_model=config.mllm_model,
                api_key=config.mllm_api_key,
                api_base_url=config.mllm_base_url,
                api_service_name=config.mllm_service_name,
            )
        )
    pipeline = EvaluationPipeline(
        device=config.device,
        enable_mllm=config.enable_mllm,
        mllm_client=mllm_client,
        video_config=config.video_config,
        parallel=config.parallel,
        max_workers=config.max_workers,
        au_backend=config.au_backend,
        au_external_python=config.au_external_python,
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


def run_analysis_with_hub(config: WebUIRunConfig) -> tuple[Any, float, Any]:
    mllm_client = None
    if config.enable_mllm:
        mllm_client = MLLMClient(
            MLLMConfig.from_env_with_overrides(
                backend="api",
                api_provider=config.mllm_provider,
                api_model=config.mllm_model,
                api_key=config.mllm_api_key,
                api_base_url=config.mllm_base_url,
                api_service_name=config.mllm_service_name,
            )
        )
    pipeline = EvaluationPipeline(
        device=config.device,
        enable_mllm=config.enable_mllm,
        mllm_client=mllm_client,
        video_config=config.video_config,
        parallel=config.parallel,
        max_workers=config.max_workers,
        au_backend=config.au_backend,
        au_external_python=config.au_external_python,
    )
    start_time = time.perf_counter()
    if config.scope == "anomaly":
        report, hub = pipeline.detect_anomalies_with_hub(
            config.video_path,
            anomaly_types=config.selected_dimensions,
            parallel=config.parallel,
            max_workers=config.max_workers,
        )
    else:
        report, hub = pipeline.evaluate_with_hub(
            config.video_path,
            selected_dimensions=config.selected_dimensions,
            parallel=config.parallel,
            max_workers=config.max_workers,
        )
    return report, time.perf_counter() - start_time, hub


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
