from unittest.mock import patch, MagicMock

import numpy as np

from src.feature_hub.extractors.face_embedding import extract_face_embeddings


def test_extract_face_embeddings_structure():
    with patch(
        "src.feature_hub.extractors.face_embedding._load_frames"
    ) as mock_load:
        mock_load.return_value = [np.zeros((100, 100, 3), dtype=np.uint8)]
        with patch(
            "src.feature_hub.extractors.face_embedding._get_face_analyzer"
        ) as mock_fa:
            mock_face = MagicMock()
            mock_face.embedding = np.random.rand(512).astype(np.float32)
            mock_face.bbox = np.array([10, 10, 50, 50])
            mock_face.det_score = 0.95
            mock_analyzer = MagicMock()
            mock_analyzer.get.return_value = [mock_face]
            mock_fa.return_value = mock_analyzer
            result = extract_face_embeddings("test.mp4", "cpu")
            assert isinstance(result, list)
            assert len(result) == 1
            assert "faces" in result[0]
