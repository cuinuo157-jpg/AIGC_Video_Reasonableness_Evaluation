import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from unittest.mock import MagicMock

from src.feature_hub.hub import FeatureHub, VideoProcessingConfig


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


def test_hub_keeps_video_config():
    cfg = VideoProcessingConfig(sample_stride=2, max_frames=16, max_side=512)
    hub = FeatureHub(video_path="test.mp4", device="cpu", video_config=cfg)
    assert hub.video_config == cfg


def test_hub_keeps_runtime_options():
    runtime_options = {
        "au_backend": "subprocess",
        "au_external_python": "D:/envs/pyfeat/python.exe",
    }
    hub = FeatureHub(video_path="test.mp4", device="cpu", runtime_options=runtime_options)
    assert hub.runtime_options == runtime_options


def test_hub_parallel_get_only_computes_once():
    hub = FeatureHub(video_path="test.mp4", device="cpu")
    call_count = 0

    def extractor(video_path, device):
        nonlocal call_count
        call_count += 1
        time.sleep(0.05)
        return {"video_path": video_path, "device": device}

    hub.register_extractor("shared", extractor)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(hub.get, "shared") for _ in range(4)]
        results = [future.result() for future in futures]

    assert call_count == 1
    assert all(result == results[0] for result in results)
