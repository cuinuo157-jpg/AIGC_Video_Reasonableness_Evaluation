import pytest
from unittest.mock import patch, MagicMock

import numpy as np

from src.mllm.client import MLLMClient
from src.mllm.config import MLLMConfig


def test_mllm_client_init():
    config = MLLMConfig(backend="api", api_provider="openai", api_model="gpt-4o")
    client = MLLMClient(config)
    assert client.config.backend == "api"


def test_mllm_client_api_call():
    config = MLLMConfig(
        backend="api", api_provider="openai", api_model="gpt-4o", api_key="test"
    )
    client = MLLMClient(config)
    frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
    with patch.object(client, "_call_api", return_value={"is_natural": True}):
        result = client.judge_video_clip(frames, "Is this natural?")
        assert result == {"is_natural": True}


def test_mllm_client_fallback():
    config = MLLMConfig(
        backend="hybrid",
        local_model="test",
        api_provider="openai",
        api_model="gpt-4o",
        api_key="test",
    )
    client = MLLMClient(config)
    frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
    with patch.object(client, "_call_local", side_effect=RuntimeError("GPU OOM")):
        with patch.object(client, "_call_api", return_value={"fallback": True}):
            result = client.judge_with_fallback(frames, "test prompt")
            assert result == {"fallback": True}


def test_mllm_client_dashscope_video_call():
    config = MLLMConfig(
        backend="api",
        api_provider="dashscope",
        api_model="qwen3-vl-8b-thinking",
        api_key="test_key",
    )
    client = MLLMClient(config)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.output = {
        "choices": [
            {
                "message": {
                    "content": [{"text": '{"is_reasonable": true, "overall_score": 8}'}]
                }
            }
        ]
    }
    mock_mm = MagicMock()
    mock_mm.call.return_value = mock_response

    with patch.object(client, "_ensure_dashscope", return_value=(MagicMock(), mock_mm)):
        result = client.judge_video_path("D:/tmp/video.mp4", "分析该视频是否存在异常", fps=2)

    assert result == {"is_reasonable": True, "overall_score": 8}
    mock_mm.call.assert_called_once()
    kwargs = mock_mm.call.call_args.kwargs
    assert kwargs["api_key"] == "test_key"
    assert kwargs["model"] == "qwen3-vl-8b-thinking"
    assert kwargs["messages"][0]["content"][0]["video"] == "file://D:/tmp/video.mp4"
    assert kwargs["messages"][0]["content"][0]["fps"] == 2
