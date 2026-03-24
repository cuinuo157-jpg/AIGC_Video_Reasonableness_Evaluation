import numpy as np
from unittest.mock import MagicMock, patch

from src.temporal_coherence.analyzer import TemporalCoherenceAnalyzer
from src.temporal_coherence.config import TemporalCoherenceConfig


def _make_hub(n_frames: int = 12) -> MagicMock:
    hub = MagicMock()
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(n_frames)]
    hub.get.side_effect = lambda k: {"video_frames": frames}.get(k, [])
    hub.device = "cpu"
    return hub


def test_temporal_coherence_detects_abnormal_disappear():
    analyzer = TemporalCoherenceAnalyzer(
        TemporalCoherenceConfig(sample_interval=5, iou_match_threshold=0.3)
    )
    hub = _make_hub(n_frames=16)

    # 采样帧: 0, 5, 10, 15
    # 仅在 0、5 有检测，且在中间位置消失 -> abnormal disappear
    fake_dets = {
        0: [np.array([30, 30, 60, 60], dtype=np.float32)],
        5: [np.array([31, 31, 61, 61], dtype=np.float32)],
        10: [],
        15: [],
    }

    with patch("src.temporal_coherence.analyzer._extract_boxes_with_gdino", return_value=fake_dets):
        result = analyzer.analyze(hub)

    assert result.applicable is True
    assert len(result.temporal_events) >= 1
    assert any(e.event_type == "disappear" for e in result.temporal_events)
    assert any(e.reason == "abnormal" for e in result.temporal_events)
    assert result.temporal_coherence_score < 1.0


def test_temporal_coherence_edge_emerge_is_not_abnormal():
    analyzer = TemporalCoherenceAnalyzer(
        TemporalCoherenceConfig(
            sample_interval=5,
            edge_margin_ratio=0.1,
            iou_match_threshold=0.2,
        )
    )
    hub = _make_hub(n_frames=16)

    # 采样帧: 0, 5, 10, 15
    # 5 帧开始从左边缘进入 -> edge_emerge
    fake_dets = {
        0: [],
        5: [np.array([0, 35, 22, 70], dtype=np.float32)],
        10: [np.array([12, 36, 35, 72], dtype=np.float32)],
        15: [np.array([20, 36, 42, 72], dtype=np.float32)],
    }

    with patch("src.temporal_coherence.analyzer._extract_boxes_with_gdino", return_value=fake_dets):
        result = analyzer.analyze(hub)

    assert result.applicable is True
    assert any(e.reason == "edge_emerge" for e in result.temporal_events)
    assert all(e.reason != "abnormal" for e in result.temporal_events)
    assert result.temporal_coherence_score == 1.0
