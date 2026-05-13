"""Pydantic request/response schemas for the FastAPI evaluation service."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Request ──────────────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    """Request body for POST /api/evaluate."""

    video_path: str | None = Field(
        default=None,
        description="本地视频路径（与 video_dir 二选一）",
        examples=["data/sample.mp4"],
    )
    video_dir: str | None = Field(
        default=None,
        description="视频目录路径（批量模式，与 video_path 二选一）",
        examples=["data/videos"],
    )
    scope: str = Field(
        default="anomaly",
        description="分析范围: 'anomaly'（五类异常）或 'full'（全量维度）",
        pattern=r"^(anomaly|full)$",
    )
    anomaly_types: list[str] | None = Field(
        default=None,
        description="五类异常模式下的维度选择",
        examples=[["face_identity", "expression", "motion_logic"]],
    )
    selected_dimensions: list[str] | None = Field(
        default=None,
        description="全量维度模式下的维度选择",
    )
    device: str = Field(default="cuda", description="推理设备")
    au_backend: str = Field(default="subprocess", pattern=r"^(local|subprocess)$")
    au_external_python: str | None = Field(
        default=None,
        description="Py-Feat 外部 Python 路径",
    )
    enable_mllm: bool = Field(default=False, description="是否启用 MLLM/VLM 判定")
    mllm_provider: str = Field(
        default="vllm",
        description="MLLM 提供方",
        examples=["vllm", "dashscope", "openai", "anthropic", "huawei_custom"],
    )
    mllm_model: str | None = Field(default=None, description="MLLM 模型名")
    mllm_base_url: str | None = Field(default=None, description="MLLM API 地址")
    mllm_api_key: str | None = Field(default=None, description="MLLM API 密钥")
    mllm_service_name: str | None = Field(
        default=None, description="huawei_custom 的 service_name"
    )
    sample_stride: int = Field(default=2, ge=1, description="采样步长")
    max_frames: int | None = Field(default=48, ge=2, description="最大帧数")
    max_side: int | None = Field(default=640, ge=64, description="最大边长")
    parallel: bool = Field(default=True, description="是否启用并发检测")
    max_workers: int | None = Field(default=None, ge=1, description="最大并发数")
    file_extensions: str | None = Field(
        default=".mp4,.avi,.mov,.mkv,.webm",
        description="批量模式下扫描的视频扩展名（逗号分隔）",
    )
    recursive_scan: bool = Field(
        default=False, description="批量模式下是否递归扫描子目录"
    )


# ── Response ─────────────────────────────────────────────────────────

class EvaluateAccepted(BaseModel):
    """Response when a job is accepted for async processing."""

    job_id: str = Field(description="任务 ID，用于轮询状态和日志")
    status: str = Field(description="任务状态: 'queued'")
    video_name: str = Field(description="视频文件名或目录名")


class JobSnapshot(BaseModel):
    """Snapshot of a job's current state."""

    job_id: str
    status: str
    created_at: float
    updated_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    result_json_path: str | None = None
    log_path: str | None = None
    artifact_root: str | None = None
    has_result: bool = False
    result: dict[str, Any] | None = None


class JobLogs(BaseModel):
    """Streaming log lines for a job."""

    job_id: str
    offset: int
    next_offset: int
    lines: list[str]
    completed: bool


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"


class DimensionInfo(BaseModel):
    """Metadata about a single evaluation dimension."""

    key: str
    label: str
    description: str
    scope: str


class ScopeInfo(BaseModel):
    """Metadata about an evaluation scope."""

    key: str
    label: str
    description: str
    dimensions: list[DimensionInfo]


class APIConfig(BaseModel):
    """API discoverability/config endpoint."""

    scopes: list[ScopeInfo]
    mllm_providers: list[str]
    defaults: dict[str, Any]
