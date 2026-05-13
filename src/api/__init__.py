"""Shared core logic for AIGC Video Reasonableness Evaluation.

Contains AnalysisConfig, Job, JobManager, analysis runners, scanning,
reporting, and dimension summarizers. Used by both WebUI and FastAPI server.
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

__all__ = [
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
