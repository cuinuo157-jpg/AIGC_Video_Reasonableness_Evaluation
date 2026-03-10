import pytest
from unittest.mock import MagicMock

from src.feature_hub.hub import FeatureHub


def test_hub_init():
    hub = FeatureHub(video_path="test.mp4", device="cpu")
    assert hub.video_path == "test.mp4"
    assert hub.device == "cpu"


def test_hub_register_and_get_extractor():
    hub = FeatureHub(video_path="test.mp4", device="cpu")
    mock_extractor = MagicMock(return_value={"data": [1, 2, 3]})
    hub.register_extractor("test_feature", mock_extractor)
    result = hub.get("test_feature")
    assert result == {"data": [1, 2, 3]}
    mock_extractor.assert_called_once()


def test_hub_caches_result():
    hub = FeatureHub(video_path="test.mp4", device="cpu")
    call_count = 0

    def extractor(video_path, device):
        nonlocal call_count
        call_count += 1
        return {"data": "result"}

    hub.register_extractor("feat", extractor)
    hub.get("feat")
    hub.get("feat")
    assert call_count == 1


def test_hub_unknown_feature_raises():
    hub = FeatureHub(video_path="test.mp4", device="cpu")
    with pytest.raises(KeyError):
        hub.get("unknown_feature")


def test_hub_available_features():
    hub = FeatureHub(video_path="test.mp4", device="cpu")
    hub.register_extractor("a", lambda vp, d: None)
    hub.register_extractor("b", lambda vp, d: None)
    assert set(hub.available_features()) == {"a", "b"}
