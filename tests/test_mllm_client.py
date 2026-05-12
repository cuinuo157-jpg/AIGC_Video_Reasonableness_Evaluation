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


def test_mllm_client_vllm_video_path_parses_json():
    config = MLLMConfig(
        backend="api",
        api_provider="vllm",
        api_model="qwen3.5:9b",
        api_key=None,
        vllm_max_frames=2,
    )
    client = MLLMClient(config)

    with patch(
        "src.mllm.vllm_openai_video.extract_frames_jpeg_bytes",
        return_value=[b"jpeg1", b"jpeg2"],
    ):
        with patch(
            "src.mllm.vllm_openai_video.chat_completions_text",
            return_value='{"ok": true, "score": 7}',
        ) as mock_chat:
            result = client.judge_video_path("fake.mp4", "prompt", fps=2)

    assert result == {"ok": True, "score": 7}
    mock_chat.assert_called_once()
    kw = mock_chat.call_args.kwargs
    assert kw["model"] == "qwen3.5:9b"
    assert kw["timeout"] == 300.0


def test_mllm_client_vllm_clip_no_response_format():
    config = MLLMConfig(
        backend="api",
        api_provider="vllm",
        api_model="qwen3.5:9b",
        vllm_max_frames=8,
    )
    client = MLLMClient(config)
    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]

    mock_choice = MagicMock()
    mock_choice.message.content = '{"x": 1}'
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_openai_cls.return_value = mock_client
        out = client.judge_video_clip(frames, "hi")

    assert out == {"x": 1}
    mock_client.chat.completions.create.assert_called_once()
    create_kw = mock_client.chat.completions.create.call_args.kwargs
    assert "response_format" not in create_kw


def test_mllm_client_huawei_custom_clip_call():
    config = MLLMConfig(
        backend="api",
        api_provider="huawei_custom",
        api_model="Qwen3-VL-32B-Instruct",
        api_base_url="http://aitest-beta.rnd.huawei.com/v1",
        api_service_name="simple_client",
        api_system_prompt="你是一位专业的 AI 助手",
        vllm_timeout=45,
    )
    client = MLLMClient(config)
    frames = [np.zeros((12, 12, 3), dtype=np.uint8) for _ in range(2)]

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "code": 0,
        "message": '{"has_violations": false, "physics_score": 0.91}',
        "request_id": "abc123",
    }

    with patch("requests.post", return_value=mock_response) as mock_post:
        out = client.judge_video_clip(frames, "请判断该视频是否违反物理常识")

    assert out == {"has_violations": False, "physics_score": 0.91}
    mock_response.raise_for_status.assert_called_once()
    mock_post.assert_called_once()
    kwargs = mock_post.call_args.kwargs
    assert kwargs["json"]["model_type"] == "Qwen3-VL-32B-Instruct"
    assert kwargs["json"]["service_name"] == "simple_client"
    assert kwargs["json"]["system_content"] == "你是一位专业的 AI 助手"
    assert kwargs["timeout"] == 45.0
    assert kwargs["headers"]["Content-Type"] == "application/json"
    user_content = kwargs["json"]["user_content"]
    assert user_content[-1]["type"] == "text"
    assert user_content[-1]["text"] == "请判断该视频是否违反物理常识"
    assert user_content[0]["type"] == "image_url"


def test_mllm_client_huawei_custom_parses_nested_message_dict():
    config = MLLMConfig(
        backend="api",
        api_provider="huawei_custom",
        api_model="Qwen3-VL-32B-Instruct",
        api_base_url="http://aitest-beta.rnd.huawei.com/v1",
    )
    client = MLLMClient(config)
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "code": 0,
        "data": {
            "message": {
                "summary": "正常",
                "issues": [],
                "is_reasonable": True,
            }
        },
    }

    with patch("requests.post", return_value=mock_response):
        out = client.judge_video_clip(frames, "请分析")

    assert out["is_reasonable"] is True
