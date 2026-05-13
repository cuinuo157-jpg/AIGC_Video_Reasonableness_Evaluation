"""WebUI reporting — thin adapter over src.api.core reporting functions.

All dimension summarizers, card builders, and report builders live in src.api.core.
This module re-exports them for backward compatibility and adds the WebUIRunConfig
import alias needed by existing callers.
"""

from __future__ import annotations

from src.api.core import (
    build_batch_report,
    build_dashboard_report,
)
from .service import WebUIRunConfig

__all__ = [
    "build_batch_report",
    "build_dashboard_report",
]
