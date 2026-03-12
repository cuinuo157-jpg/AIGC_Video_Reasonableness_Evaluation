"""测试 biological_anomaly analyzer 三级编排。"""

from unittest.mock import MagicMock

from src.biological_anomaly.analyzer import BiologicalAnomalyAnalyzer


def _make_hub(has_faces: bool = True, n_frames: int = 5):
    """构建 mock FeatureHub。"""
    hub = MagicMock()

    face_data = []
    keypoints = []
    frames = []

    for i in range(n_frames):
        if has_faces:
            face_data.append({"faces": [{"bbox": [0, 0, 50, 50]}], "num_faces": 1})
        else:
            face_data.append({"faces": [], "num_faces": 0})

        keypoints.append(
            {
                "body": None,
                "left_hand": None,
                "right_hand": None,
                "face": None,
                "ear_left": 0.30,
                "ear_right": 0.30,
                "ear_avg": 0.30,
                "mar": 0.15,
                "frame_idx": i,
            }
        )
        import numpy as np

        frames.append(np.zeros((100, 100, 3), dtype=np.uint8))

    def side_effect(key):
        return {
            "face_embedding": face_data,
            "keypoints": keypoints,
            "video_frames": frames,
        }[key]

    hub.get.side_effect = side_effect
    return hub


def test_bio_analyzer_no_faces():
    hub = _make_hub(has_faces=False)
    result = BiologicalAnomalyAnalyzer().analyze(hub)
    assert result.applicable is False


def test_bio_analyzer_returns_score():
    hub = _make_hub(has_faces=True, n_frames=10)
    result = BiologicalAnomalyAnalyzer().analyze(hub)
    assert result.applicable is True
    assert 0.0 <= result.bio_quality_score <= 1.0


def test_bio_analyzer_three_level_scores():
    hub = _make_hub(has_faces=True, n_frames=10)
    result = BiologicalAnomalyAnalyzer().analyze(hub)
    assert hasattr(result, "level1_score")
    assert hasattr(result, "level2_score")
    assert hasattr(result, "level3_score")


def test_bio_analyzer_mllm_disabled():
    hub = _make_hub(has_faces=True)
    analyzer = BiologicalAnomalyAnalyzer(mllm_client=None)
    result = analyzer.analyze(hub)
    assert result.mllm_anomalies == []
    assert result.level3_score == 1.0


def test_bio_analyzer_with_mock_mllm():
    hub = _make_hub(has_faces=True, n_frames=5)
    mock_mllm = MagicMock()
    mock_mllm.judge_video_clip.return_value = {
        "has_anomalies": False,
        "anomalies": [],
    }

    from src.biological_anomaly.config import BiologicalAnomalyConfig

    config = BiologicalAnomalyConfig(enable_mllm=True)
    analyzer = BiologicalAnomalyAnalyzer(config=config, mllm_client=mock_mllm)
    result = analyzer.analyze(hub)
    assert result.applicable is True
