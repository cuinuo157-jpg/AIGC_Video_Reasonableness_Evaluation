from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from src.evaluation_pipeline import DimensionResult, EvaluationReport
from src.mllm.config import MLLMConfig
from src.webui.reporting import build_dashboard_report
from src.webui import service as webui_service
from src.webui.service import build_frontend_config, build_run_config


def test_build_frontend_config_contains_scopes():
    config = build_frontend_config()
    scopes = {item["key"] for item in config["scopes"]}
    assert scopes == {"anomaly", "full"}
    assert "anomaly_types" in config["defaults"]
    assert config["defaults"]["au_backend"] == "subprocess"
    assert config["defaults"]["au_external_python"]
    assert config["defaults"]["mllm_provider"] == MLLMConfig.api_provider
    assert "huawei_custom" in config["mllm_providers"]


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
    assert config.au_backend == "subprocess"
    assert config.au_external_python
    assert config.mllm_provider == MLLMConfig.api_provider
    assert config.mllm_model == MLLMConfig.api_model
    assert "physics" in config.selected_dimensions


def test_build_run_config_accepts_au_routing_values(tmp_path: Path):
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"fake")

    config = build_run_config(
        {
            "video_path": str(video_path),
            "scope": "anomaly",
            "au_backend": "local",
            "au_external_python": "D:/custom/pyfeat/python.exe",
        }
    )

    assert config.au_backend == "local"
    assert config.au_external_python == "D:/custom/pyfeat/python.exe"


def test_build_run_config_accepts_mllm_values(tmp_path: Path):
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"fake")

    config = build_run_config(
        {
            "video_path": str(video_path),
            "scope": "anomaly",
            "enable_mllm": "true",
            "mllm_provider": "huawei_custom",
            "mllm_model": "Qwen3-VL-32B-Instruct",
            "mllm_base_url": "http://aitest-beta.rnd.huawei.com/v1",
            "mllm_api_key": "test-key",
            "mllm_service_name": "simple_client",
        }
    )

    assert config.enable_mllm is True
    assert config.mllm_provider == "huawei_custom"
    assert config.mllm_model == "Qwen3-VL-32B-Instruct"
    assert config.mllm_base_url == "http://aitest-beta.rnd.huawei.com/v1"
    assert config.mllm_api_key == "test-key"
    assert config.mllm_service_name == "simple_client"


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
    assert payload["video_processing"]["mllm_provider"] == run_config.mllm_provider
    assert payload["video_processing"]["mllm_model"] == run_config.mllm_model
    assert payload["video_processing"]["au_backend"] == run_config.au_backend
    assert payload["video_processing"]["au_external_python"] == run_config.au_external_python


def test_run_analysis_builds_mllm_client(tmp_path: Path, monkeypatch):
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"fake")
    run_config = build_run_config(
        {
            "video_path": str(video_path),
            "scope": "anomaly",
            "enable_mllm": "true",
            "mllm_provider": "huawei_custom",
            "mllm_model": "Qwen3-VL-32B-Instruct",
            "mllm_base_url": "http://aitest-beta.rnd.huawei.com/v1",
            "mllm_api_key": "test-key",
            "mllm_service_name": "simple_client",
        }
    )

    captured = {}

    class DummyPipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def detect_anomalies(self, *_args, **_kwargs):
            return EvaluationReport(
                dimensions={},
                active_dimensions=[],
                final_score=0.0,
            )

    monkeypatch.setattr(webui_service, "EvaluationPipeline", DummyPipeline)

    report, elapsed = webui_service.run_analysis(run_config)

    assert isinstance(report, EvaluationReport)
    assert elapsed >= 0
    assert captured["enable_mllm"] is True
    assert captured["mllm_client"] is not None
    assert captured["mllm_client"].config.api_provider == "huawei_custom"
    assert captured["mllm_client"].config.api_model == "Qwen3-VL-32B-Instruct"
    assert captured["mllm_client"].config.api_base_url == "http://aitest-beta.rnd.huawei.com/v1"
    assert captured["mllm_client"].config.api_service_name == "simple_client"


def test_job_manager_streams_logs_and_persists_outputs(tmp_path: Path, monkeypatch):
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"fake")
    run_config = build_run_config(
        {
            "video_path": str(video_path),
            "scope": "anomaly",
            "anomaly_types": ["face_identity"],
            "enable_mllm": "true",
            "mllm_provider": "huawei_custom",
            "mllm_model": "Qwen3-VL-32B-Instruct",
            "mllm_base_url": "http://aitest-beta.rnd.huawei.com/v1",
            "mllm_service_name": "simple_client",
        }
    )

    def fake_run_analysis(_config):
        print("stdout ready")
        sys.stderr.write("stderr ready\n")
        logging.getLogger("webui-test").warning("warning ready")
        report = EvaluationReport(
            dimensions={
                "face_identity": DimensionResult(
                    applicable=True,
                    score=0.88,
                    weight=1.0,
                    details=SimpleNamespace(
                        identity_score=0.88,
                        csim_ref=0.91,
                        csim_adj=0.9,
                        csim_min=0.86,
                        face_tracks=[object()],
                        drop_events=[],
                    ),
                )
            },
            active_dimensions=["face_identity"],
            final_score=0.88,
        )
        return report, 0.12

    monkeypatch.setattr(webui_service, "run_analysis", fake_run_analysis)

    manager = webui_service.WebUIJobManager(results_dir=tmp_path / "results")
    job = manager.create_job(run_config)

    snapshot = None
    for _ in range(60):
        snapshot = manager.get_job_snapshot(job.job_id)
        if snapshot["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)

    assert snapshot is not None
    assert snapshot["status"] == "completed"
    assert snapshot["result"] is not None
    assert snapshot["result_json_path"] is not None
    assert snapshot["log_path"] is not None

    result_path = Path(snapshot["result_json_path"])
    log_path = Path(snapshot["log_path"])
    assert result_path.exists()
    assert log_path.exists()

    all_logs = manager.get_job_logs(job.job_id, offset=0)
    assert any("stdout ready" in line for line in all_logs["lines"])
    assert any("stderr ready" in line for line in all_logs["lines"])
    assert any("warning ready" in line for line in all_logs["lines"])
    assert any("AU 路由" in line for line in all_logs["lines"])
    assert any("MLLM:" in line for line in all_logs["lines"])

    sliced_logs = manager.get_job_logs(job.job_id, offset=2)
    assert sliced_logs["offset"] == 2
    assert sliced_logs["next_offset"] == all_logs["next_offset"]
    assert len(sliced_logs["lines"]) == max(0, len(all_logs["lines"]) - 2)

    payload_text = result_path.read_text(encoding="utf-8")
    log_text = log_path.read_text(encoding="utf-8")
    assert "result_json_path" in payload_text
    assert "log_path" in payload_text
    assert "stdout ready" in log_text
