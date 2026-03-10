from unittest.mock import MagicMock, patch

import numpy as np

from src.evaluation_pipeline import (
    DimensionResult,
    EvaluationPipeline,
    EvaluationReport,
    _redistribute_weights,
)


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
