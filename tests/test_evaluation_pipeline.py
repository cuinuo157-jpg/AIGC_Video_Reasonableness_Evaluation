from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from src.evaluation_pipeline import (
    DEFAULT_ANOMALY_TYPES,
    DimensionResult,
    EvaluationPipeline,
    EvaluationReport,
    _redistribute_weights,
)
from src.feature_hub.hub import VideoProcessingConfig
from src.mllm.config import MLLMConfig
from src.mllm.client import MLLMClient


def test_pipeline_skips_inapplicable():
    pipeline = EvaluationPipeline(enable_mllm=False)
    with patch.object(pipeline, "_create_hub") as mock_hub_fn:
        hub = MagicMock()
        hub.get.side_effect = lambda k: {
            "face_embedding": [{"faces": [], "num_faces": 0}] * 5,
            "optical_flow": [(np.ones((50, 50)), np.ones((50, 50)))] * 5,
            "video_frames": [np.zeros((50, 50, 3), dtype=np.uint8)] * 5,
        }.get(k, [])
        mock_hub_fn.return_value = hub
        report = pipeline.evaluate("test.mp4")
        assert isinstance(report, EvaluationReport)
        assert report.dimensions["face_identity"].applicable is False
        assert report.dimensions["background"].applicable is True
        assert len(report.active_dimensions) >= 1


def test_pipeline_weight_redistribution():
    results = {
        "d1": DimensionResult(applicable=False, score=None, weight=0.2),
        "d2": DimensionResult(applicable=True, score=0.8, weight=0.3),
        "d3": DimensionResult(applicable=True, score=0.6, weight=0.5),
    }
    active, final = _redistribute_weights(results)
    assert len(active) == 2
    assert abs(sum(r.weight for r in active.values()) - 1.0) < 0.01
    # final = (0.3/0.8)*0.8 + (0.5/0.8)*0.6 = 0.3 + 0.375 = 0.675
    assert 0.5 < final < 0.8


def test_pipeline_enable_mllm_propagates_to_analyzers():
    mllm_client = MLLMClient(
        MLLMConfig(
            backend="api",
            api_provider="dashscope",
            api_model="qwen3-vl-8b-thinking",
            api_key="test",
        )
    )
    pipeline = EvaluationPipeline(enable_mllm=True, mllm_client=mllm_client)

    motion = pipeline._analyzers["motion_logic"]
    physics = pipeline._analyzers["physics"]
    bio = pipeline._analyzers["biological_anomaly"]

    assert motion.config.enable_mllm is True
    assert physics.config.enable_mllm is True
    assert bio.config.enable_mllm is True
    assert motion._mllm_client is mllm_client
    assert physics._mllm_client is mllm_client
    assert bio._mllm_client is mllm_client


def test_pipeline_disable_mllm_turns_off_analyzer_mllm():
    mllm_client = MLLMClient(
        MLLMConfig(
            backend="api",
            api_provider="dashscope",
            api_model="qwen3-vl-8b-thinking",
            api_key="test",
        )
    )
    pipeline = EvaluationPipeline(enable_mllm=False, mllm_client=mllm_client)

    motion = pipeline._analyzers["motion_logic"]
    physics = pipeline._analyzers["physics"]
    bio = pipeline._analyzers["biological_anomaly"]

    assert motion.config.enable_mllm is False
    assert physics.config.enable_mllm is False
    assert bio.config.enable_mllm is False


def test_detect_anomalies_only_runs_requested_types():
    pipeline = EvaluationPipeline(enable_mllm=False)
    hub = MagicMock()

    called: list[str] = []

    def make_analyzer(name: str, score_attr: str, score: float):
        analyzer = MagicMock()

        def analyze(_hub):
            called.append(name)
            return SimpleNamespace(
                applicable=True,
                skip_reason=None,
                **{score_attr: score},
            )

        analyzer.analyze.side_effect = analyze
        return analyzer

    pipeline._analyzers = {
        "face_identity": make_analyzer("face_identity", "identity_score", 0.9),
        "expression": make_analyzer("expression", "expression_score", 0.8),
        "biological_anomaly": make_analyzer("biological_anomaly", "bio_quality_score", 0.7),
        "motion_logic": make_analyzer("motion_logic", "motion_logic_score", 0.6),
        "physics": make_analyzer("physics", "physics_score", 0.5),
        "background": make_analyzer("background", "background_score", 0.4),
        "temporal_coherence": make_analyzer("temporal_coherence", "temporal_coherence_score", 0.3),
        "perceptual_quality": make_analyzer("perceptual_quality", "perceptual_quality_score", 0.2),
    }

    with patch.object(pipeline, "_create_hub", return_value=hub):
        report = pipeline.detect_anomalies(
            "test.mp4",
            anomaly_types=["identity", "motion", "physics"],
        )

    assert isinstance(report, EvaluationReport)
    assert called == ["face_identity", "motion_logic", "physics"]
    assert set(report.dimensions.keys()) == {"face_identity", "motion_logic", "physics"}


def test_detect_anomalies_defaults_to_five_types():
    pipeline = EvaluationPipeline(enable_mllm=False)
    with patch.object(pipeline, "evaluate", return_value=EvaluationReport()) as mock_evaluate:
        pipeline.detect_anomalies("test.mp4")

    _, kwargs = mock_evaluate.call_args
    assert tuple(kwargs["selected_dimensions"]) == DEFAULT_ANOMALY_TYPES


def test_pipeline_passes_video_config_to_hub_factory():
    cfg = VideoProcessingConfig(sample_stride=3, max_frames=24, max_side=640)
    pipeline = EvaluationPipeline(
        video_config=cfg,
        au_backend="subprocess",
        au_external_python="D:/envs/pyfeat/python.exe",
    )

    with patch("src.evaluation_pipeline.create_default_hub") as mock_create_default_hub:
        pipeline._create_hub("test.mp4")

    mock_create_default_hub.assert_called_once_with(
        "test.mp4",
        pipeline.device,
        video_config=cfg,
        runtime_options={
            "au_backend": "subprocess",
            "au_external_python": "D:/envs/pyfeat/python.exe",
        },
    )
