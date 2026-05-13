"""FastAPI-based AIGC Video Reasonableness Evaluation API.

Provides a production-ready REST API for single and batch video evaluation.
Exposes Swagger docs at /docs and async job management with log streaming.
"""

from .core import (
    AnalysisConfig,
    DIMENSION_CATALOG,
    Job,
    JobManager,
    build_batch_report,
    build_dashboard_report,
    parse_analysis_config,
    run_analysis,
    scan_video_directory,
)
from .server import app

__all__ = [
    "app",
    "AnalysisConfig",
    "DIMENSION_CATALOG",
    "Job",
    "JobManager",
    "build_batch_report",
    "build_dashboard_report",
    "parse_analysis_config",
    "run_analysis",
    "scan_video_directory",
]
