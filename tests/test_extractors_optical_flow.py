from unittest.mock import patch, MagicMock

import numpy as np

from src.feature_hub.extractors.optical_flow import extract_optical_flow


def test_extract_optical_flow_returns_list():
    fake_frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
    with patch(
        "src.feature_hub.extractors.optical_flow._load_frames",
        return_value=fake_frames,
    ):
        with patch(
            "src.feature_hub.extractors.optical_flow._compute_flows"
        ) as mock_compute:
            mock_compute.return_value = [
                (np.zeros((100, 100)), np.zeros((100, 100))) for _ in range(2)
            ]
            result = extract_optical_flow("test.mp4", "cpu")
            assert isinstance(result, list)
            assert len(result) == 2
            assert isinstance(result[0], tuple)
