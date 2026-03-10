import numpy as np
from unittest.mock import MagicMock

from src.face_identity.analyzer import FaceIdentityAnalyzer


def test_analyzer_with_faces():
    hub = MagicMock()
    base = np.random.rand(512).astype(np.float32)
    base /= np.linalg.norm(base)

    def _make_face():
        e = base + np.random.randn(512).astype(np.float32) * 0.01
        e /= np.linalg.norm(e)
        return e

    hub.get.return_value = [
        {
            "faces": [{"embedding": _make_face(), "bbox": [0, 0, 50, 50], "det_score": 0.9}],
            "num_faces": 1,
        }
        for _ in range(10)
    ]
    result = FaceIdentityAnalyzer().analyze(hub)
    assert result.applicable is True
    assert result.identity_score > 0.5


def test_analyzer_no_faces():
    hub = MagicMock()
    hub.get.return_value = [{"faces": [], "num_faces": 0} for _ in range(10)]
    result = FaceIdentityAnalyzer().analyze(hub)
    assert result.applicable is False
