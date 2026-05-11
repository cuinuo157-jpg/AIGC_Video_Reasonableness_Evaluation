import numpy as np
from unittest.mock import MagicMock

from src.expression_naturalness.au_extractor import AUExtractor, _patch_scipy_compat


def test_au_extractor_returns_au_dict():
    extractor = AUExtractor()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    mock_det = MagicMock()
    mock_det.detect_faces.return_value = [[1, 2, 3, 4]]
    mock_det.detect_landmarks.return_value = [np.zeros((68, 2), dtype=np.float32)]
    mock_det.detect_aus.return_value = np.array([[0.5, 1.2, 0.3]])
    mock_det.info = {"au_presence_columns": ["AU01", "AU02", "AU04"]}
    extractor._detector = mock_det

    result = extractor.extract(frame)
    assert isinstance(result, dict)
    assert result["AU01"] == 0.5
    assert result["AU02"] == 1.2


def test_patch_scipy_compat_keeps_existing_simps():
    import scipy.integrate as integrate

    original = getattr(integrate, "simps", None)
    _patch_scipy_compat()
    if original is not None:
        assert integrate.simps is original
    else:
        assert hasattr(integrate, "simps")
