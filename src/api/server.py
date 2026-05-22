"""FastAPI server for the AIGC Video Reasonableness Evaluation API.

Provides:
  POST /api/evaluate        — submit a single or batch analysis job
  GET  /api/jobs/{job_id}   — poll job status + result
  GET  /api/jobs/{job_id}/logs — stream job logs
  GET  /health              — health check
  GET  /api/config          — dimension catalog & defaults
  GET  /docs                — Swagger UI (auto)

Usage:
  uvicorn src.api.server:app --host 0.0.0.0 --port 8000
  python scripts/run_api.py
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .core import (
    DEFAULT_ANOMALY_TYPES,
    DEFAULT_AU_BACKEND,
    DEFAULT_AU_EXTERNAL_PYTHON,
    DEFAULT_DEVICE,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MAX_SIDE,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SAMPLE_STRIDE,
    DEFAULT_TOP5_TYPES,
    DEFAULT_UPLOAD_DIR,
    DIMENSION_CATALOG,
    FULL_DIMENSIONS,
    JobManager,
    SUPPORTED_MLLM_PROVIDERS,
    _default_mllm_config,
    _safe_stem,
    available_dimensions,
    parse_analysis_config,
)
from .models import (
    APIConfig,
    DimensionInfo,
    EvaluateAccepted,
    EvaluateRequest,
    HealthResponse,
    JobLogs,
    JobSnapshot,
    ScopeInfo,
)

logger = logging.getLogger(__name__)

# ── Lifespan ─────────────────────────────────────────────────────────
_job_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    assert _job_manager is not None, "JobManager not initialized"
    return _job_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _job_manager
    _job_manager = JobManager(results_dir=DEFAULT_RESULTS_DIR)
    logger.info("JobManager initialized, results_dir=%s", _job_manager.results_dir)
    yield
    logger.info("Shutting down API server")


# ── App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="AIGC Video Reasonableness Evaluation API",
    description="多维度 AIGC 视频合理性评测服务。支持单视频 + 批量目录分析，异步任务 + 日志流。",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Exception handler ────────────────────────────────────────────────

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": str(exc), "status": 400},
    )


@app.exception_handler(KeyError)
async def key_error_handler(request: Request, exc: KeyError) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"error": f"job not found: {exc}", "status": 404},
    )


# ── Routes ───────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config", response_model=APIConfig)
def get_config() -> dict[str, Any]:
    mllm_defaults = _default_mllm_config()
    return {
        "scopes": [
            {
                "key": "anomaly",
                "label": "五类异常",
                "description": "身份、表情、生物异常、运动逻辑、物理常识。",
                "dimensions": [
                    DimensionInfo(
                        key=key,
                        label=DIMENSION_CATALOG[key]["label"],
                        description=DIMENSION_CATALOG[key]["description"],
                        scope="anomaly",
                    )
                    for key in DEFAULT_ANOMALY_TYPES
                ],
            },
            {
                "key": "top5",
                "label": "Top5 场景",
                "description": "身份、生物异常、运动逻辑、物理常识、时间一致性。",
                "dimensions": [
                    DimensionInfo(
                        key=key,
                        label=DIMENSION_CATALOG[key]["label"],
                        description=DIMENSION_CATALOG[key]["description"],
                        scope="top5",
                    )
                    for key in DEFAULT_TOP5_TYPES
                ],
            },
            {
                "key": "full",
                "label": "全量维度",
                "description": "包含时间一致性、背景一致性与感知质量。",
                "dimensions": [
                    DimensionInfo(
                        key=key,
                        label=DIMENSION_CATALOG[key]["label"],
                        description=DIMENSION_CATALOG[key]["description"],
                        scope=DIMENSION_CATALOG[key].get("scope", "full"),
                    )
                    for key in FULL_DIMENSIONS
                ],
            },
        ],
        "mllm_providers": list(SUPPORTED_MLLM_PROVIDERS),
        "defaults": {
            "device": DEFAULT_DEVICE,
            "parallel": True,
            "sample_stride": DEFAULT_SAMPLE_STRIDE,
            "max_frames": DEFAULT_MAX_FRAMES,
            "max_side": DEFAULT_MAX_SIDE,
            "au_backend": DEFAULT_AU_BACKEND,
            "au_external_python": DEFAULT_AU_EXTERNAL_PYTHON,
            "enable_mllm": False,
            "mllm_provider": mllm_defaults.api_provider,
            "mllm_model": mllm_defaults.api_model,
            "mllm_base_url": mllm_defaults.api_base_url or "",
            "mllm_api_key": "",
            "mllm_service_name": mllm_defaults.api_service_name,
            "anomaly_types": list(DEFAULT_ANOMALY_TYPES),
            "selected_dimensions": list(FULL_DIMENSIONS),
        },
    }


@app.post("/api/evaluate", response_model=EvaluateAccepted, status_code=202)
def evaluate(request: EvaluateRequest) -> dict[str, Any]:
    """Submit a video analysis job (single or batch).

    Returns a job_id immediately. Poll GET /api/jobs/{job_id} for results.
    """
    mgr = get_job_manager()
    payload = request.dict(exclude_none=True)
    config = parse_analysis_config(payload)
    job = mgr.create_job(config)
    video_name = (
        Path(config.video_path).name if config.video_path else (config.video_dir or "batch")
    )
    return {
        "job_id": job.job_id,
        "status": job.status,
        "video_name": video_name,
    }


@app.get("/api/jobs/{job_id}", response_model=JobSnapshot)
def get_job(job_id: str) -> dict[str, Any]:
    """Poll job status and get results when completed."""
    mgr = get_job_manager()
    return mgr.get_job_snapshot(job_id)


@app.get("/api/jobs/{job_id}/logs", response_model=JobLogs)
def get_job_logs(job_id: str, offset: int = Query(0, ge=0)) -> dict[str, Any]:
    """Streaming log endpoint. Poll with increasing offset for real-time logs."""
    mgr = get_job_manager()
    return mgr.get_job_logs(job_id, offset=offset)


@app.post("/api/evaluate/upload", response_model=EvaluateAccepted, status_code=202)
async def evaluate_upload(
    file: UploadFile = File(..., description="视频文件"),
    scope: str = Form(default="anomaly"),
    device: str = Form(default="cuda"),
    enable_mllm: bool = Form(default=False),
    mllm_provider: str = Form(default="huawei_custom"),
    mllm_model: str = Form(default=""),
    mllm_base_url: str = Form(default=""),
    mllm_api_key: str = Form(default=""),
    mllm_service_name: str = Form(default=""),
    parallel: bool = Form(default=True),
    max_workers: int | None = Form(default=None),
    sample_stride: int = Form(default=2),
    max_frames: int | None = Form(default=None),
    max_side: int | None = Form(default=None),
    save_visualizations: bool = Form(default=False),
) -> dict[str, Any]:
    """Submit analysis via file upload. Saves file server-side then processes."""
    mgr = get_job_manager()

    upload_dir = DEFAULT_UPLOAD_DIR
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "upload.mp4").name.replace(" ", "_")
    stem = _safe_stem(safe_name, max_len=80)
    suffix = Path(safe_name).suffix or ".mp4"
    dest = upload_dir / f"{int(time.time())}_{stem}{suffix}"
    dest.write_bytes(await file.read())

    payload = {
        "video_path": os.fspath(dest),
        "scope": scope,
        "device": device,
        "enable_mllm": enable_mllm,
        "mllm_provider": mllm_provider,
        "mllm_model": mllm_model or None,
        "mllm_base_url": mllm_base_url or None,
        "mllm_api_key": mllm_api_key or None,
        "mllm_service_name": mllm_service_name or None,
        "parallel": parallel,
        "max_workers": max_workers,
        "sample_stride": sample_stride,
        "max_frames": max_frames,
        "max_side": max_side,
        "save_visualizations": save_visualizations,
    }
    config = parse_analysis_config(payload)
    job = mgr.create_job(config)
    return {
        "job_id": job.job_id,
        "status": job.status,
        "video_name": dest.name,
    }


@app.get("/api/jobs/{job_id}/artifacts")
def download_artifacts(job_id: str):
    """Download all generated files (reports, visualizations) as a zip archive."""
    mgr = get_job_manager()
    job = mgr.get_job(job_id)
    # 使用 get_job_snapshot 读取状态（内部加锁），避免跨线程读到 stale status
    snapshot = mgr.get_job_snapshot(job_id)
    if snapshot["status"] not in ("completed", "failed"):
        raise HTTPException(400, "任务未完成，无法下载产物")

    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Result JSON
        if job.result:
            zf.writestr("result.json", json.dumps(job.result, ensure_ascii=False, indent=2))

        # Log file
        if job.log_path:
            lp = Path(job.log_path)
            if lp.exists():
                zf.write(lp, f"logs/{lp.name}")

        # Per-video reports (batch)
        if isinstance(job.result, dict) and job.result.get("batch"):
            for vr in job.result.get("video_results", []):
                rp = vr.get("report_path")
                if rp:
                    p = Path(rp)
                    if p.exists():
                        zf.write(p, f"reports/{p.name}")

        # Artifact directory
        if job.artifact_root:
            art_root = Path(job.artifact_root)
            if art_root.is_dir():
                for f in art_root.rglob("*"):
                    if f.is_file():
                        zf.write(f, f"visualizations/{f.relative_to(art_root)}")

    buf.seek(0)
    name = _safe_stem(job.run_config.video_path or "result", max_len=40)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}_artifacts.zip"'},
    )


@app.post("/api/jobs/{job_id}/cleanup")
def cleanup_job(job_id: str) -> dict[str, Any]:
    """Delete all temporary files for a completed job."""
    mgr = get_job_manager()
    # 使用 get_job_snapshot 读取状态（内部加锁），避免跨线程读到 stale status
    snapshot = mgr.get_job_snapshot(job_id)
    if snapshot["status"] not in ("completed", "failed"):
        raise HTTPException(400, "任务未完成，无法清理")
    job = mgr.get_job(job_id)
    deleted: list[str] = []

    # Delete uploaded video
    vp = job.run_config.video_path
    if vp:
        try:
            p = Path(vp)
            if p.exists():
                p.unlink()
                deleted.append(str(p))
        except Exception:
            pass

    # Delete result JSON
    if job.result_json_path:
        try:
            p = Path(job.result_json_path)
            if p.exists():
                p.unlink()
                deleted.append(str(p))
        except Exception:
            pass

    # Delete log file
    if job.log_path:
        try:
            p = Path(job.log_path)
            if p.exists():
                p.unlink()
                deleted.append(str(p))
        except Exception:
            pass

    # Delete artifact directory
    if job.artifact_root:
        try:
            import shutil
            p = Path(job.artifact_root)
            if p.is_dir():
                shutil.rmtree(p)
                deleted.append(str(p))
        except Exception:
            pass

    # Delete per-video reports for batch
    if isinstance(job.result, dict) and job.result.get("batch"):
        for vr in job.result.get("video_results", []):
            rp = vr.get("report_path")
            if rp:
                try:
                    p = Path(rp)
                    if p.exists():
                        p.unlink()
                        deleted.append(str(p))
                except Exception:
                    pass

    return {"deleted": deleted, "count": len(deleted)}
