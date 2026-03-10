import numpy as np
from unittest.mock import MagicMock

from src.expression_naturalness.au_extractor import AUExtractor


def test_au_extractor_returns_au_dict():
    extractor = AUExtractor()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    mock_result = MagicMock()
    mock_result.aus = MagicMock()
    mock_result.aus.__len__ = MagicMock(return_value=1)
    mock_result.aus.values = [[0.5, 1.2, 0.3]]
    mock_result.aus.columns = ["AU01", "AU02", "AU04"]

    mock_det = MagicMock()
    mock_det.detect_image.return_value = mock_result
    extractor._detector = mock_det

    result = extractor.extract(frame)
    assert isinstance(result, dict)
    assert "AU01" in result
