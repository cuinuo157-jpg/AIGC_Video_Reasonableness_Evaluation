from unittest.mock import patch, MagicMock

import numpy as np

from src.feature_hub.extractors.depth import extract_depth_maps


def test_extract_depth_maps_returns_list():
    with patch("src.feature_hub.extractors.depth._load_frames") as mock_load:
        mock_load.return_value = [
            np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)
        ]
        with patch("src.feature_hub.extractors.depth._get_depth_model") as mock_model:
            mock_pred = MagicMock()
            mock_pred.squeeze.return_value.cpu.return_value.numpy.return_value = (
                np.zeros((384, 384))
            )
            mock_m = MagicMock(return_value=mock_pred)
            mock_transform = MagicMock(
                return_value=MagicMock(
                    to=MagicMock(return_value="tensor")
                )
            )
            mock_model.return_value = (mock_m, mock_transform)
            result = extract_depth_maps("test.mp4", "cpu")
            assert isinstance(result, list)
            assert len(result) == 3
