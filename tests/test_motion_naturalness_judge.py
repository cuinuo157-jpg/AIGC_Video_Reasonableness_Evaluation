from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from src.mllm.prompts import MOTION_NATURALNESS_PROMPT
from src.motion_logic.naturalness_judge import judge_naturalness_mllm


def test_judge_naturalness_mllm_uses_video_path_for_dashscope():
    hub = MagicMock()
    hub.video_path = "D:/tmp/demo.mp4"
    mllm_client = MagicMock()
    mllm_client.config = SimpleNamespace(api_provider="dashscope")
    mllm_client.judge_video_path.return_value = {"is_reasonable": False, "issues": ["jump cut"]}

    result = judge_naturalness_mllm(
        hub=hub,
        mllm_client=mllm_client,
        flows=[(np.zeros((4, 4)), np.zeros((4, 4)))],
        smoothness_score=0.3,
    )

    assert result["is_reasonable"] is False
    mllm_client.judge_video_path.assert_called_once_with("D:/tmp/demo.mp4", MOTION_NATURALNESS_PROMPT)
    mllm_client.judge_video_clip.assert_not_called()


def test_judge_naturalness_mllm_uses_video_path_for_vllm():
    hub = MagicMock()
    hub.video_path = "D:/tmp/demo.mp4"
    mllm_client = MagicMock()
    mllm_client.config = SimpleNamespace(api_provider="vllm")
    mllm_client.judge_video_path.return_value = {"is_reasonable": True}

    result = judge_naturalness_mllm(
        hub=hub,
        mllm_client=mllm_client,
        flows=[(np.zeros((4, 4)), np.zeros((4, 4)))],
        smoothness_score=0.3,
    )

    assert result["is_reasonable"] is True
    mllm_client.judge_video_path.assert_called_once_with("D:/tmp/demo.mp4", MOTION_NATURALNESS_PROMPT)


def test_judge_naturalness_mllm_falls_back_to_frames_for_non_dashscope():
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
    hub = MagicMock()
    hub.get.side_effect = lambda k: frames if k == "video_frames" else KeyError(k)
    mllm_client = MagicMock()
    mllm_client.config = SimpleNamespace(api_provider="openai")
    mllm_client.judge_video_clip.return_value = {"is_natural": True, "issues": []}

    result = judge_naturalness_mllm(
        hub=hub,
        mllm_client=mllm_client,
        flows=[(np.zeros((4, 4)), np.zeros((4, 4)))],
        smoothness_score=0.3,
    )

    assert result["is_natural"] is True
    mllm_client.judge_video_clip.assert_called_once()


def test_judge_naturalness_mllm_falls_back_to_frames_for_huawei_custom():
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
    hub = MagicMock()
    hub.video_path = "D:/tmp/demo.mp4"
    hub.get.side_effect = lambda k: frames if k == "video_frames" else KeyError(k)
    mllm_client = MagicMock()
    mllm_client.config = SimpleNamespace(api_provider="huawei_custom")
    mllm_client.judge_video_clip.return_value = {"is_reasonable": True, "issues": []}

    result = judge_naturalness_mllm(
        hub=hub,
        mllm_client=mllm_client,
        flows=[(np.zeros((4, 4)), np.zeros((4, 4)))],
        smoothness_score=0.3,
    )

    assert result["is_reasonable"] is True
    mllm_client.judge_video_clip.assert_called_once()
    mllm_client.judge_video_path.assert_not_called()
