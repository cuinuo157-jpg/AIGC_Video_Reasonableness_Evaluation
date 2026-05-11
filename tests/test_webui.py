from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.evaluation_pipeline import DimensionResult, EvaluationReport
from src.webui.reporting import build_dashboard_report
from src.webui.service import build_frontend_config, build_run_config


def test_build_frontend_config_contains_scopes():
    config = build_frontend_config()
    scopes = {item["key"] for item in config["scopes"]}
    assert scopes == {"anomaly", "full"}
    assert "anomaly_types" in config["defaults"]


def test_build_run_config_uses_scope_defaults(tmp_path: Path):
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"fake")

    config = build_run_config(
        {
            "video_path": str(video_path),
            "scope": "anomaly",
            "parallel": "true",
            "sample_stride": "3",
            "max_frames": "24",
            "max_side": "512",
        }
    )

    assert config.scope == "anomaly"
    assert config.video_path == str(video_path)
    assert config.parallel is True
    assert config.video_config.sample_stride == 3
    assert config.video_config.max_frames == 24
    assert config.video_config.max_side == 512
    assert "physics" in config.selected_dimensions


def test_build_dashboard_report_summarizes_dimension_cards(tmp_path: Path):
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"fake")
    run_config = build_run_config(
        {
            "video_path": str(video_path),
            "scope": "anomaly",
            "anomaly_types": ["face_identity", "physics"],
        }
    )

    report = EvaluationReport(
        dimensions={
            "face_identity": DimensionResult(
                applicable=True,
                score=0.91,
                weight=0.5,
                details=SimpleNamespace(
                    identity_score=0.91,
                    csim_ref=0.94,
                    csim_adj=0.92,
                    csim_min=0.88,
                    face_tracks=[object()],
                    drop_events=[],
                ),
            ),
            "physics": DimensionResult(
                applicable=False,
                skip_reason="no motion",
                score=None,
                weight=0.5,
                details=SimpleNamespace(),
            ),
        },
        active_dimensions=["face_identity"],
        final_score=0.91,
    )

    payload = build_dashboard_report(report, run_config, elapsed_sec=1.25)

    assert payload["video_name"] == "demo.mp4"
    assert payload["summary"]["applicable_count"] == 1
    assert payload["summary"]["skipped_count"] == 1
    assert payload["dimensions"][0]["label"] == "身份一致性"
    assert payload["dimensions"][0]["metrics"][0]["label"] == "身份分"
    assert payload["dimensions"][1]["applicable"] is False
