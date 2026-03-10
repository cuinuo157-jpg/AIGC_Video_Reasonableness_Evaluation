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
