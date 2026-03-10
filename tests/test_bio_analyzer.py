from unittest.mock import MagicMock

from src.biological_anomaly.analyzer import BiologicalAnomalyAnalyzer


def test_bio_analyzer_no_faces():
    hub = MagicMock()
    hub.get.return_value = [{"faces": [], "num_faces": 0} for _ in range(5)]
    result = BiologicalAnomalyAnalyzer().analyze(hub)
    assert result.applicable is False


def test_bio_analyzer_returns_score():
    hub = MagicMock()
    hub.get.side_effect = lambda key: {
        "face_embedding": [
            {"faces": [{"bbox": [0, 0, 50, 50]}], "num_faces": 1} for _ in range(5)
        ]
    }.get(key, [])
    result = BiologicalAnomalyAnalyzer().analyze(hub)
    assert 0.0 <= result.bio_quality_score <= 1.0
