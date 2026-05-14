"""Shared core logic for both webui and api services.

Extracted from src/webui/service.py and src/webui/reporting.py.
Contains job management, analysis execution, video scanning, and reporting.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation_pipeline import (
    DEFAULT_ANOMALY_TYPES,
    DEFAULT_WEIGHTS,
    DimensionResult,
    EvaluationPipeline,
    EvaluationReport,
)
from src.expression_naturalness.au_extractor import (
    _DEFAULT_AU_BACKEND as DEFAULT_AU_BACKEND,
    _DEFAULT_AU_PYTHON as DEFAULT_AU_EXTERNAL_PYTHON,
)
from src.feature_hub import VideoProcessingConfig
from src.mllm import MLLMClient, MLLMConfig

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════

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
DEFAULT_RESULTS_DIR = Path("outputs") / "api_results"
DEFAULT_VISUALIZATION_DIR = Path("outputs") / "webui_artifacts"
MAX_LOG_LINES = 800
SUPPORTED_MLLM_PROVIDERS = ("vllm", "dashscope", "openai", "anthropic", "huawei_custom")
BATCH_VIDEO_EXTENSIONS = (
    ".mp4", ".avi", ".mov", ".mkv", ".webm",
    ".MP4", ".AVI", ".MOV", ".MKV", ".WEBM",
)


# ═══════════════════════════════════════════════════════════════════
#  Utility
# ═══════════════════════════════════════════════════════════════════

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
    value: Any, default: int | None, *,
    minimum: int | None = None, allow_none: bool = False,
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


# ═══════════════════════════════════════════════════════════════════
#  Dataclasses
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for a single or batch analysis job."""
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
class Job:
    """Tracks state of a single (or batch) analysis job."""
    job_id: str
    run_config: AnalysisConfig
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


# ═══════════════════════════════════════════════════════════════════
#  Log writers (internal)
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
#  JobManager
# ═══════════════════════════════════════════════════════════════════

class JobManager:
    """Manages analysis jobs with background thread execution."""

    def __init__(self, results_dir: Path | None = None) -> None:
        self.results_dir = (results_dir or DEFAULT_RESULTS_DIR).resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()

    def create_job(self, run_config: AnalysisConfig) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id=job_id, run_config=run_config)
        with self._lock:
            self._jobs[job_id] = job
        self._append_log(job_id, f"[job] 已创建任务 {job_id}")
        worker = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        worker.start()
        return job

    def get_job(self, job_id: str) -> Job:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    def get_job_snapshot(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        with self._lock:
            status = job.status
            # Override if batch results show all videos done
            if status == "running" and isinstance(job.result, dict) and job.result.get("batch"):
                total = job.result.get("total_videos", 0)
                done = (job.result.get("completed_videos", 0) +
                        job.result.get("failed_videos", 0))
                if total > 0 and done >= total:
                    status = "completed"
            return {
                "job_id": job.job_id,
                "status": status,
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
            terminal = job.status in {"completed", "failed"}
            # Also treat as complete if batch result shows all videos done
            if not terminal and isinstance(job.result, dict) and job.result.get("batch"):
                total = job.result.get("total_videos", 0)
                done = (job.result.get("completed_videos", 0) +
                        job.result.get("failed_videos", 0))
                if total > 0 and done >= total:
                    terminal = True
            return {
                "job_id": job.job_id,
                "offset": safe_offset,
                "next_offset": len(job.logs),
                "lines": lines,
                "completed": terminal,
            }

    def _append_log(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.logs.append(message)
            if len(job.logs) > MAX_LOG_LINES:
                del job.logs[:len(job.logs) - MAX_LOG_LINES]
            job.updated_at = time.time()

    def _build_output_paths(self, job: Job) -> tuple[Path, Path]:
        stem = Path(job.run_config.video_path).stem[:80] or "video"
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(job.created_at))
        result_path = self.results_dir / f"{timestamp}_{stem}_{job.job_id}_report.json"
        log_path = self.results_dir / f"{timestamp}_{stem}_{job.job_id}.log"
        return result_path, log_path

    def _build_artifact_dir(self, job: Job) -> Path:
        stem = Path(job.run_config.video_path).stem[:80] or "video"
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(job.created_at))
        base_dir = Path(job.run_config.visualization_root or DEFAULT_VISUALIZATION_DIR).resolve()
        return base_dir / f"{timestamp}_{stem}_{job.job_id}"

    def _persist_job_outputs(
        self, job: Job,
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

    def _run_batch_job(self, job: Job, video_list: list[str], append_fn: Any) -> None:
        """Run batch analysis sequentially for all videos in the list.

        Creates a single EvaluationPipeline instance so models are loaded once
        and reused across all videos. Only the FeatureHub is recreated per video.
        """
        total = len(video_list)
        batch_results: list[dict[str, Any]] = []
        batch_start = time.perf_counter()

        writer = _JobLogWriter(append_fn)
        tee_stdout = _TeeWriter(sys.stdout, writer)
        tee_stderr = _TeeWriter(sys.stderr, writer)
        root_logger = logging.getLogger()
        log_handler = _JobLogHandler(append_fn)
        root_logger.addHandler(log_handler)

        # ── Build shared pipeline (models loaded once) ──
        mllm_client = None
        if job.run_config.enable_mllm:
            mllm_client = MLLMClient(MLLMConfig.from_env_with_overrides(
                backend="api", api_provider=job.run_config.mllm_provider,
                api_model=job.run_config.mllm_model,
                api_key=job.run_config.mllm_api_key,
                api_base_url=job.run_config.mllm_base_url,
                api_service_name=job.run_config.mllm_service_name,
            ))
        pipeline = EvaluationPipeline(
            device=job.run_config.device,
            enable_mllm=job.run_config.enable_mllm,
            mllm_client=mllm_client,
            video_config=job.run_config.video_config,
            parallel=job.run_config.parallel,
            max_workers=job.run_config.max_workers,
            au_backend=job.run_config.au_backend,
            au_external_python=job.run_config.au_external_python,
        )
        is_anomaly = job.run_config.scope == "anomaly"

        try:
            for idx, video_path in enumerate(video_list, 1):
                video_name = Path(video_path).name
                append_fn(f"[batch] 处理 {idx}/{total}: {video_name}")

                # Per-video config (metadata only, pipeline already created)
                single_config = AnalysisConfig(
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
                    save_visualizations=job.run_config.save_visualizations,
                    visualization_root=job.run_config.visualization_root,
                    parallel=job.run_config.parallel,
                    max_workers=job.run_config.max_workers,
                    video_config=job.run_config.video_config,
                )

                try:
                    append_fn(f"[batch] {idx}/{total}: 开始抽帧与分析 {video_name}")
                    t0 = time.perf_counter()
                    with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
                        if is_anomaly:
                            if single_config.save_visualizations:
                                report, hub = pipeline.detect_anomalies_with_hub(
                                    video_path, anomaly_types=single_config.selected_dimensions,
                                    parallel=single_config.parallel, max_workers=single_config.max_workers)
                            else:
                                report = pipeline.detect_anomalies(
                                    video_path, anomaly_types=single_config.selected_dimensions,
                                    parallel=single_config.parallel, max_workers=single_config.max_workers)
                                hub = None
                        else:
                            if single_config.save_visualizations:
                                report, hub = pipeline.evaluate_with_hub(
                                    video_path, selected_dimensions=single_config.selected_dimensions,
                                    parallel=single_config.parallel, max_workers=single_config.max_workers)
                            else:
                                report = pipeline.evaluate(
                                    video_path, selected_dimensions=single_config.selected_dimensions,
                                    parallel=single_config.parallel, max_workers=single_config.max_workers)
                                hub = None
                    elapsed = time.perf_counter() - t0
                    writer.flush()
                    result_data = build_dashboard_report(report, single_config, elapsed)

                    # Extract VLM raw outputs per dimension
                    vlm_outputs: dict[str, Any] = {}
                    for card in result_data.get("dimensions", []):
                        if card.get("vlm_raw_output") is not None:
                            vlm_outputs[card["key"]] = card["vlm_raw_output"]

                    # Generate visual artifacts if enabled
                    artifacts = []
                    artifact_root = None
                    if single_config.save_visualizations and hub is not None:
                        stem = video_name.rsplit(".", 1)[0][:80] or "video"
                        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(job.created_at))
                        base_dir = Path(single_config.visualization_root or DEFAULT_VISUALIZATION_DIR).resolve()
                        artifact_dir = base_dir / f"{ts}_{stem}_{job.job_id}"
                        try:
                            from src.webui.artifacts import generate_visual_artifacts
                            artifact_root, artifacts, _ = generate_visual_artifacts(
                                report, single_config, hub, artifact_dir, log_fn=append_fn)
                        except Exception:
                            pass

                    # Save per-video report
                    stem = Path(video_path).stem[:80] or "video"
                    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(job.created_at))
                    report_path = self.results_dir / f"{ts}_{stem}_{job.job_id}_per_video.json"
                    report_path.write_text(
                        json.dumps(result_data, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                    batch_results.append({
                        "video_name": video_name,
                        "video_path": video_path,
                        "final_score": result_data.get("final_score"),
                        "status": "completed",
                        "elapsed_sec": round(elapsed, 3),
                        "active_dimensions": result_data.get("active_dimensions", []),
                        "vlm_outputs": vlm_outputs,
                        "report_path": os.fspath(report_path),
                        "artifact_root": artifact_root,
                        "error": None,
                    })
                    append_fn(
                        f"[batch] ✓ {idx}/{total}: {video_name} - "
                        f"综合分 {result_data.get('final_score', 0):.3f} ({elapsed:.1f}s)"
                    )
                except Exception as exc:
                    writer.flush()
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

                # Free GPU memory between videos
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception:
                    pass
                gc.collect()

                with self._lock:
                    job.result = {
                        "batch": True,
                        "batch_progress": {"current": idx, "total": total, "current_video": video_name},
                        "video_dir": job.run_config.video_dir,
                        "total_videos": total,
                        "completed_videos": len([r for r in batch_results if r["status"] == "completed"]),
                        "failed_videos": len([r for r in batch_results if r["status"] == "failed"]),
                        "video_results": list(batch_results),
                    }
        finally:
            root_logger.removeHandler(log_handler)

        total_elapsed = time.perf_counter() - batch_start
        final_report = build_batch_report(batch_results, job.run_config, total_elapsed,
                                            results_dir=os.fspath(self.results_dir))

        # Set status BEFORE file I/O so frontend sees result even if writes fail
        with self._lock:
            # Collect artifact roots for the artifact_root_path display
            roots = [r.get("artifact_root") for r in batch_results if r.get("artifact_root")]
            if roots:
                # Use common parent directory as batch artifact root
                job.artifact_root = os.fspath(Path(roots[0]).parent)
            job.result = final_report
            job.status = "completed"
            job.completed_at = time.time()
            job.updated_at = job.completed_at

        result_path, log_path = self._build_output_paths(job)
        job.result_json_path = os.fspath(result_path)
        job.log_path = os.fspath(log_path)

        result_path.write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")
        log_path.write_text("\n".join(job.logs), encoding="utf-8")

        completed = final_report["completed_videos"]
        failed = final_report["failed_videos"]
        avg = final_report["aggregate"]["avg_score"]
        append_fn(
            f"[batch] 批量处理完成: {total} 个视频, "
            f"成功 {completed}, 失败 {failed}, 平均分 {avg:.3f}, 总耗时 {total_elapsed:.1f}s"
        )

    def _run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        with self._lock:
            job.status = "running"
            job.updated_at = time.time()

        def append(message: str) -> None:
            self._append_log(job_id, message)

        try:
            append(f"[job] 开始分析 {job.run_config.video_path or job.run_config.video_dir}")
            append(f"[job] 范围={job.run_config.scope}, "
                   f"维度={','.join(job.run_config.selected_dimensions)}, "
                   f"device={job.run_config.device}, parallel={job.run_config.parallel}")
            append(f"[job] AU 路由: backend={job.run_config.au_backend}, "
                   f"external_python={job.run_config.au_external_python or '-'}")
            append(f"[job] 可视化产物: {'开启' if job.run_config.save_visualizations else '关闭'}, "
                   f"root={job.run_config.visualization_root or '-'}")
            if job.run_config.enable_mllm:
                append(f"[job] MLLM: provider={job.run_config.mllm_provider}, "
                       f"model={job.run_config.mllm_model}, "
                       f"base_url={job.run_config.mllm_base_url or '-'}, "
                       f"service_name={job.run_config.mllm_service_name or '-'}")

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
                try:
                    self._run_batch_job(job, video_list, append)
                finally:
                    # Safety net: ensure status is terminal if batch method didn't set it
                    if job.status == "running":
                        with self._lock:
                            if job.status == "running":
                                job.status = "completed" if job.result else "failed"
                                job.completed_at = time.time()
                                job.updated_at = job.completed_at
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

                writer.flush()
                result_path, log_path = self._build_output_paths(job)
                result_json_path = os.fspath(result_path)
                log_path_str = os.fspath(log_path)
                payload = build_dashboard_report(report, job.run_config, elapsed)

                if job.run_config.save_visualizations and artifact_dir is not None and hub is not None:
                    from src.webui.artifacts import generate_visual_artifacts
                    artifact_root, artifacts, artifacts_by_dimension = generate_visual_artifacts(
                        report, job.run_config, hub, artifact_dir, log_fn=append)
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
                logger.exception("job failed: %s", job_id)
            finally:
                root_logger.removeHandler(log_handler)

        except Exception as exc:
            append(f"[job] 任务启动失败: {exc}")
            logger.exception("job startup failed: %s", job_id)
            with self._lock:
                job.status = "failed"
                job.error = str(exc)
                job.completed_at = time.time()
                job.updated_at = job.completed_at
            try:
                _, log_path = self._build_output_paths(job)
                job.log_path = os.fspath(log_path)
                log_path.write_text("\n".join(job.logs), encoding="utf-8")
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════
#  Config parser
# ═══════════════════════════════════════════════════════════════════

def parse_analysis_config(
    payload: dict[str, Any],
    uploaded_video_path: str | None = None,
) -> AnalysisConfig:
    """Build AnalysisConfig from a dict payload (form data or JSON body)."""
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
    mllm_model = str(payload.get("mllm_model", mllm_defaults.api_model)).strip() or mllm_defaults.api_model
    mllm_base_url = str(payload.get("mllm_base_url", mllm_defaults.api_base_url or "")).strip() or None
    mllm_api_key = (
        str(payload.get("mllm_api_key", mllm_defaults.api_key or "")).strip()
        or mllm_defaults.api_key or None
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
    file_extensions = (
        str(payload.get("file_extensions", ".mp4,.avi,.mov,.mkv,.webm")).strip()
        or ".mp4,.avi,.mov,.mkv,.webm"
    )
    recursive_scan = _coerce_bool(payload.get("recursive_scan"), False)

    return AnalysisConfig(
        video_path=video_path, scope=scope, selected_dimensions=selected_dimensions,
        device=device, au_backend=au_backend, au_external_python=au_external_python,
        enable_mllm=enable_mllm, mllm_provider=mllm_provider, mllm_model=mllm_model,
        mllm_base_url=mllm_base_url, mllm_api_key=mllm_api_key, mllm_service_name=mllm_service_name,
        save_visualizations=save_visualizations, visualization_root=visualization_root,
        parallel=_coerce_bool(payload.get("parallel"), True), max_workers=max_workers,
        video_config=video_config, video_dir=video_dir,
        file_extensions=file_extensions, recursive_scan=recursive_scan,
    )


# ═══════════════════════════════════════════════════════════════════
#  Analysis runners
# ═══════════════════════════════════════════════════════════════════

def run_analysis(config: AnalysisConfig) -> tuple[Any, float]:
    """Run evaluation pipeline for a single video. Returns (report, elapsed_sec)."""
    mllm_client = None
    if config.enable_mllm:
        mllm_client = MLLMClient(MLLMConfig.from_env_with_overrides(
            backend="api", api_provider=config.mllm_provider, api_model=config.mllm_model,
            api_key=config.mllm_api_key, api_base_url=config.mllm_base_url,
            api_service_name=config.mllm_service_name,
        ))
    pipeline = EvaluationPipeline(
        device=config.device, enable_mllm=config.enable_mllm, mllm_client=mllm_client,
        video_config=config.video_config, parallel=config.parallel, max_workers=config.max_workers,
        au_backend=config.au_backend, au_external_python=config.au_external_python,
    )
    start_time = time.perf_counter()
    if config.scope == "anomaly":
        report = pipeline.detect_anomalies(
            config.video_path, anomaly_types=config.selected_dimensions,
            parallel=config.parallel, max_workers=config.max_workers,
        )
    else:
        report = pipeline.evaluate(
            config.video_path, selected_dimensions=config.selected_dimensions,
            parallel=config.parallel, max_workers=config.max_workers,
        )
    return report, time.perf_counter() - start_time


def run_analysis_with_hub(config: AnalysisConfig) -> tuple[Any, float, Any]:
    """Run evaluation, returning (report, elapsed_sec, feature_hub)."""
    mllm_client = None
    if config.enable_mllm:
        mllm_client = MLLMClient(MLLMConfig.from_env_with_overrides(
            backend="api", api_provider=config.mllm_provider, api_model=config.mllm_model,
            api_key=config.mllm_api_key, api_base_url=config.mllm_base_url,
            api_service_name=config.mllm_service_name,
        ))
    pipeline = EvaluationPipeline(
        device=config.device, enable_mllm=config.enable_mllm, mllm_client=mllm_client,
        video_config=config.video_config, parallel=config.parallel, max_workers=config.max_workers,
        au_backend=config.au_backend, au_external_python=config.au_external_python,
    )
    start_time = time.perf_counter()
    if config.scope == "anomaly":
        report, hub = pipeline.detect_anomalies_with_hub(
            config.video_path, anomaly_types=config.selected_dimensions,
            parallel=config.parallel, max_workers=config.max_workers,
        )
    else:
        report, hub = pipeline.evaluate_with_hub(
            config.video_path, selected_dimensions=config.selected_dimensions,
            parallel=config.parallel, max_workers=config.max_workers,
        )
    return report, time.perf_counter() - start_time, hub


# ═══════════════════════════════════════════════════════════════════
#  Reporting helpers
# ═══════════════════════════════════════════════════════════════════

def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except Exception:
        return None

def _m(label: str, value: Any, kind: str = "number") -> dict[str, Any]:
    return {"label": label, "value": value, "kind": kind}

def _ev(title: str, detail: str, severity: str = "info") -> dict[str, str]:
    return {"title": title, "detail": detail, "severity": severity}

def _jr(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jr(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jr(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {key: _jr(getattr(value, key)) for key in value.__dataclass_fields__.keys()}
    return str(value)

def _top(items: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    return items[:limit]

def _band(score: float | None) -> str:
    if score is None:
        return "na"
    if score >= 0.85:
        return "excellent"
    if score >= 0.7:
        return "good"
    if score >= 0.5:
        return "warning"
    return "critical"


# ── Dimension summarizers ─────────────────────────────────────────

def _summarize_face_identity(raw: Any) -> tuple:
    metrics = [
        _m("身份分", _f(getattr(raw, "identity_score", None)), "score"),
        _m("参考相似度", _f(getattr(raw, "csim_ref", None)), "score"),
        _m("相邻相似度", _f(getattr(raw, "csim_adj", None)), "score"),
        _m("最低相似度", _f(getattr(raw, "csim_min", None)), "score"),
        _m("轨迹数", len(getattr(raw, "face_tracks", []) or []), "count"),
        _m("突降事件", len(getattr(raw, "drop_events", []) or []), "count"),
    ]
    drop_events = [
        _ev(f"帧 {ev.frame_idx} 身份相似度突降",
            f"{ev.similarity_before:.3f} -> {ev.similarity_after:.3f}", "warning")
        for ev in getattr(raw, "drop_events", []) or []
    ]
    highlights = [
        f"最低身份相似度 {getattr(raw, 'csim_min', 0.0):.3f}",
        f"主人脸轨迹数 {len(getattr(raw, 'face_tracks', []) or [])}",
    ]
    return metrics, highlights, _top(drop_events)

def _summarize_expression(raw: Any) -> tuple:
    violations = getattr(raw, "combination_violations", []) or []
    metrics = [
        _m("表情分", _f(getattr(raw, "expression_score", None)), "score"),
        _m("时序平滑度", _f(getattr(raw, "temporal_smoothness", None)), "score"),
        _m("AU 规则冲突", len(violations), "count"),
        _m("AU 通道数", len(getattr(raw, "au_sequences", {}) or {}), "count"),
    ]
    events = []
    for v in violations[:5]:
        rule = getattr(v, "rule_name", "规则冲突")
        reason = getattr(v, "description", "") or getattr(v, "message", "")
        events.append(_ev(rule, reason, "warning"))
    highlights = [
        f"检测到 {len(violations)} 个 AU 组合异常",
        f"表情过渡平滑度 {getattr(raw, 'temporal_smoothness', 0.0):.3f}",
    ]
    return metrics, highlights, events

def _summarize_biological(raw: Any) -> tuple:
    body_scores = getattr(raw, "body_part_scores", {}) or {}
    metrics = [
        _m("综合分", _f(getattr(raw, "bio_quality_score", None)), "score"),
        _m("Level 1", _f(getattr(raw, "level1_score", None)), "score"),
        _m("Level 2", _f(getattr(raw, "level2_score", None)), "score"),
        _m("Level 3", _f(getattr(raw, "level3_score", None)), "score"),
        _m("总异常数", getattr(raw, "anomaly_count", 0), "count"),
        _m("眼部异常", len(getattr(raw, "eye_anomalies", []) or []), "count"),
        _m("嘴部异常", len(getattr(raw, "mouth_anomalies", []) or []), "count"),
        _m("手部异常", len(getattr(raw, "hand_anomalies", []) or []), "count"),
        _m("身体异常", len(getattr(raw, "body_anomalies", []) or []), "count"),
    ]
    all_events = []
    for section_name, anomalies in (
        ("眼部", getattr(raw, "eye_anomalies", []) or []),
        ("嘴部", getattr(raw, "mouth_anomalies", []) or []),
        ("手部", getattr(raw, "hand_anomalies", []) or []),
        ("身体", getattr(raw, "body_anomalies", []) or []),
        ("MLLM", getattr(raw, "mllm_anomalies", []) or []),
    ):
        for a in anomalies:
            detail = a.get("type") or a.get("description") or a.get("reason") or "异常"
            frame_idx = a.get("frame_idx", "-")
            severity = a.get("severity", "warning")
            all_events.append(_ev(f"{section_name} / 帧 {frame_idx}", str(detail), str(severity)))
    body_hint = ", ".join(f"{n}:{s:.2f}" for n, s in list(body_scores.items())[:4]) or "暂无部位稳定度统计"
    highlights = [
        f"共定位 {getattr(raw, 'anomaly_count', 0)} 处生物特征异常",
        f"部位稳定度 {body_hint}",
    ]
    return metrics, highlights, _top(all_events)

def _summarize_motion(raw: Any) -> tuple:
    naturalness_issues = getattr(raw, "naturalness_issues", []) or []
    traj_detail = getattr(raw, "trajectory_curvature_detail", None)
    subject_detail = getattr(raw, "subject_motion_detail", None)
    metrics = [
        _m("综合分", _f(getattr(raw, "motion_logic_score", None)), "score"),
        _m("动态度", _f(getattr(raw, "dynamics_score", None)), "score"),
        _m("平滑度", _f(getattr(raw, "smoothness_score", None)), "score"),
        _m("运动自然度", _f(getattr(raw, "naturalness_score", None)), "score"),
        _m("轨迹异常数", getattr(traj_detail, "abnormal_event_count", 0) if traj_detail else 0, "count"),
        _m("主体感知运动分", _f(getattr(subject_detail, "perceptual_score", None) if subject_detail else None), "score"),
    ]
    events = [_ev("自然度问题", issue, "warning") for issue in naturalness_issues[:5]]
    if traj_detail and getattr(traj_detail, "abnormal_ratio", None) is not None:
        events.append(_ev("轨迹曲率异常", f"异常占比 {traj_detail.abnormal_ratio:.3f}", "warning"))
    highlights = [
        f"动态度 {getattr(raw, 'dynamics_score', 0.0):.3f}，平滑度 {getattr(raw, 'smoothness_score', 0.0):.3f}",
        f"自然度问题 {len(naturalness_issues)} 条",
    ]
    return metrics, highlights, _top(events)

def _summarize_physics(raw: Any) -> tuple:
    violations = getattr(raw, "vlm_violations", []) or []
    drift_events = getattr(raw, "drift_events", []) or []
    metrics = [
        _m("综合分", _f(getattr(raw, "physics_score", None)), "score"),
        _m("像素漂移分", _f(getattr(raw, "drift_score", None)), "score"),
        _m("VLM 物理分", _f(getattr(raw, "vlm_score", None)), "score"),
        _m("漂移事件", len(drift_events), "count"),
        _m("物理违规", len(violations), "count"),
    ]
    events = []
    for v in violations[:5]:
        title = v.get("type") or "物理违规"
        detail = v.get("description") or v.get("reason") or str(v)
        events.append(_ev(title, detail, "critical"))
    for d in drift_events[:3]:
        detail = f"平均幅度 {d.get('avg_magnitude', 0):.3f}, 持续 {d.get('duration_frames', 0)} 帧"
        events.append(_ev("像素漂移", detail, "warning"))
    highlights = [
        f"漂移事件 {len(drift_events)} 个，物理违规 {len(violations)} 个",
        str(getattr(raw, "vlm_reasoning", "")).strip()[:140] or "未返回额外物理推理文本",
    ]
    return metrics, highlights, _top(events)

def _summarize_background(raw: Any) -> tuple:
    metrics = [
        _m("综合分", _f(getattr(raw, "background_score", None)), "score"),
        _m("静态残差", _f(getattr(raw, "residual_score", None)), "score"),
        _m("单应稳定性", _f(getattr(raw, "homography_stability", None)), "score"),
        _m("深度一致性", _f(getattr(raw, "depth_consistency", None)), "score"),
    ]
    highlights = [
        f"背景残差 {getattr(raw, 'residual_score', 0.0):.3f}",
        f"单应稳定性 {getattr(raw, 'homography_stability', 0.0):.3f}",
    ]
    return metrics, highlights, []

def _summarize_temporal(raw: Any) -> tuple:
    abnormal_events = getattr(raw, "abnormal_events", []) or []
    temporal_events = getattr(raw, "temporal_events", []) or []
    metrics = [
        _m("综合分", _f(getattr(raw, "temporal_coherence_score", None)), "score"),
        _m("总事件数", len(temporal_events), "count"),
        _m("异常事件", len(abnormal_events), "count"),
    ]
    events = [
        _ev(f"{e.event_type} / 帧 {e.frame_idx}", f"track {e.track_id}，原因 {e.reason}", "warning")
        for e in abnormal_events[:5]
    ]
    highlights = [
        f"抽样轨迹事件 {len(temporal_events)} 个",
        f"异常出现/消失事件 {len(abnormal_events)} 个",
    ]
    return metrics, highlights, events

def _summarize_perceptual(raw: Any) -> tuple:
    frame_scores = getattr(raw, "frame_quality_scores", []) or []
    weakest_idx = int(np.argmin(frame_scores)) if frame_scores else -1
    weakest_score = float(np.min(frame_scores)) if frame_scores else 0.0
    metrics = [
        _m("综合分", _f(getattr(raw, "perceptual_quality_score", None)), "score"),
        _m("清晰度", _f(getattr(raw, "blur_score", None)), "score"),
        _m("一致性", _f(getattr(raw, "consistency_score", None)), "score"),
        _m("瑕疵分", _f(getattr(raw, "artifact_score", None)), "score"),
        _m("分析帧数", len(frame_scores), "count"),
    ]
    highlights = [
        f"最弱帧 {weakest_idx if weakest_idx >= 0 else '-'}，质量分 {weakest_score:.3f}",
        f"平均帧质量 {float(np.mean(frame_scores)):.3f}" if frame_scores else "未返回逐帧质量分",
    ]
    return metrics, highlights, []

def _summarize_generic(raw: Any, score: float | None) -> tuple:
    metrics = [_m("得分", score, "score")]
    if is_dataclass(raw):
        fields = [n for n in raw.__dataclass_fields__.keys() if n not in {"applicable", "skip_reason"}]
        highlights = [f"字段: {', '.join(fields[:5])}"]
    else:
        highlights = ["无定制摘要，已返回基础得分"]
    return metrics, highlights, []

def _extract_vlm_raw(name: str, raw: Any) -> Any:
    if name == "motion_logic":
        return _jr(getattr(raw, "naturalness_mllm_result", None))
    if name == "physics":
        return _jr(getattr(raw, "vlm_raw_result", None))
    if name == "biological_anomaly":
        return _jr(getattr(raw, "mllm_raw_result", None))
    return None

_SUMMARIZERS = {
    "face_identity": _summarize_face_identity,
    "expression": _summarize_expression,
    "biological_anomaly": _summarize_biological,
    "motion_logic": _summarize_motion,
    "physics": _summarize_physics,
    "background": _summarize_background,
    "temporal_coherence": _summarize_temporal,
    "perceptual_quality": _summarize_perceptual,
}

def build_dashboard_report(
    report: EvaluationReport, run_config: AnalysisConfig, elapsed_sec: float,
) -> dict[str, Any]:
    """Build a single-video dashboard report dict."""
    selected_order = list(run_config.selected_dimensions)
    cards = []
    for name in selected_order:
        if name not in report.dimensions:
            continue
        result = report.dimensions[name]
        label = DIMENSION_CATALOG.get(name, {}).get("label", name)
        desc = DIMENSION_CATALOG.get(name, {}).get("description", "")
        score = _f(result.score)
        if not result.applicable:
            cards.append({
                "key": name, "label": label, "description": desc,
                "applicable": False, "skip_reason": result.skip_reason or "not applicable",
                "score": score, "weight": _f(result.weight), "band": _band(score),
                "metrics": [], "highlights": [], "events": [], "vlm_raw_output": None, "artifacts": [],
            })
            continue
        raw = result.details
        summarizer = _SUMMARIZERS.get(name, lambda item: _summarize_generic(item, score))
        metrics, highlights, events = summarizer(raw)
        cards.append({
            "key": name, "label": label, "description": desc,
            "applicable": True, "skip_reason": result.skip_reason,
            "score": score, "weight": _f(result.weight), "band": _band(score),
            "metrics": metrics, "highlights": highlights, "events": events,
            "vlm_raw_output": _extract_vlm_raw(name, raw), "artifacts": [],
        })

    scored = [c for c in cards if c["score"] is not None]
    best = max(scored, key=lambda c: c["score"]) if scored else None
    worst = min(scored, key=lambda c: c["score"]) if scored else None
    return {
        "video_name": Path(run_config.video_path).name,
        "video_path": run_config.video_path,
        "scope": run_config.scope, "device": run_config.device,
        "elapsed_sec": round(elapsed_sec, 3),
        "final_score": _f(report.final_score),
        "active_dimensions": report.active_dimensions,
        "selected_dimensions": selected_order,
        "video_processing": {
            "sample_stride": run_config.video_config.sample_stride,
            "max_frames": run_config.video_config.max_frames,
            "max_side": run_config.video_config.max_side,
            "parallel": run_config.parallel, "max_workers": run_config.max_workers,
            "enable_mllm": run_config.enable_mllm,
            "mllm_provider": run_config.mllm_provider,
            "mllm_model": run_config.mllm_model,
            "mllm_base_url": run_config.mllm_base_url,
            "mllm_service_name": run_config.mllm_service_name,
            "au_backend": run_config.au_backend,
            "au_external_python": run_config.au_external_python,
            "save_visualizations": run_config.save_visualizations,
            "visualization_root": run_config.visualization_root,
        },
        "summary": {
            "dimension_count": len(cards),
            "applicable_count": sum(1 for c in cards if c["applicable"]),
            "skipped_count": sum(1 for c in cards if not c["applicable"]),
            "best_dimension": best["label"] if best else None,
            "best_score": best["score"] if best else None,
            "worst_dimension": worst["label"] if worst else None,
            "worst_score": worst["score"] if worst else None,
        },
        "dimensions": cards,
        "artifact_root": None,
        "artifacts": [],
    }


def build_batch_report(
    batch_results: list[dict[str, Any]], run_config: AnalysisConfig, total_elapsed: float,
    results_dir: str = "",
) -> dict[str, Any]:
    """Build aggregated batch report from per-video results."""
    completed = [r for r in batch_results if r["status"] == "completed"]
    failed = [r for r in batch_results if r["status"] == "failed"]
    scores = [r["final_score"] for r in completed if r["final_score"] is not None]
    avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    best = max(completed, key=lambda r: r["final_score"] or 0) if completed else None
    worst = min(completed, key=lambda r: r["final_score"] or 0) if completed else None
    return {
        "batch": True,
        "video_dir": getattr(run_config, "video_dir", None),
        "results_dir": results_dir,
        "scope": getattr(run_config, "scope", "full"),
        "device": getattr(run_config, "device", "cuda"),
        "elapsed_sec": round(total_elapsed, 3),
        "total_videos": len(batch_results),
        "completed_videos": len(completed),
        "failed_videos": len(failed),
        "video_results": batch_results,
        "aggregate": {
            "avg_score": avg_score,
            "best_video": best["video_name"] if best else None,
            "best_score": best["final_score"] if best else None,
            "worst_video": worst["video_name"] if worst else None,
            "worst_score": worst["final_score"] if worst else None,
            "total_elapsed": round(total_elapsed, 3),
        },
        "video_processing": {
            "sample_stride": run_config.video_config.sample_stride,
            "max_frames": run_config.video_config.max_frames,
            "max_side": run_config.video_config.max_side,
            "parallel": run_config.parallel,
            "max_workers": run_config.max_workers,
            "enable_mllm": run_config.enable_mllm,
            "mllm_provider": run_config.mllm_provider,
            "mllm_model": run_config.mllm_model,
        },
    }


def available_dimensions(scope: str) -> list[dict[str, str]]:
    """Get available dimension choices for a given scope."""
    keys = DEFAULT_ANOMALY_TYPES if scope == "anomaly" else FULL_DIMENSIONS
    return [{"key": key, **DIMENSION_CATALOG[key]} for key in keys]
