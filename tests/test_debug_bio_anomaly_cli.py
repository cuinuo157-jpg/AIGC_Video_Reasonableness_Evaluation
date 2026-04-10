from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import scripts.debug_bio_anomaly as debug_bio
from src.biological_anomaly.config import BiologicalAnomalyConfig


def test_build_mllm_client_disabled_returns_none():
    args = SimpleNamespace(
        mllm_provider="dashscope",
        mllm_model="qwen3-vl-8b-thinking",
        mllm_api_key="x",
        mllm_base_url="",
        mllm_fps=2,
    )
    assert debug_bio.build_mllm_client(args, enable_mllm=False) is None


def test_run_level3_skips_when_no_suspicious():
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
    keypoints_seq = [{"left_hand": None, "right_hand": None, "face": None}]
    l1 = {"all": []}
    l2 = {"all": []}
    cfg = BiologicalAnomalyConfig(enable_mllm=True)
    mllm_client = MagicMock()
    out = debug_bio.run_level3(frames, keypoints_seq, l1, l2, cfg, mllm_client)
    assert out["skipped"] is True
    assert out["level3_score"] == 1.0


def test_run_level3_calls_mllm_judge_with_suspicious():
    frames = [np.zeros((16, 16, 3), dtype=np.uint8)]
    keypoints_seq = [{"left_hand": np.random.rand(21, 3).astype(np.float32), "right_hand": None, "face": None}]
    l1 = {"all": [{"frame_idx": 0, "type": "hand_jitter", "hand": "left_hand"}]}
    l2 = {"all": []}
    cfg = BiologicalAnomalyConfig(enable_mllm=True, mllm_max_crops=4)
    mllm_client = MagicMock()
    with patch(
        "scripts.debug_bio_anomaly.judge_biological_anomaly_mllm",
        return_value={"skipped": False, "has_anomalies": True, "anomalies": [{"type": "finger_fusion"}]},
    ) as mock_judge:
        out = debug_bio.run_level3(frames, keypoints_seq, l1, l2, cfg, mllm_client)

    assert out["skipped"] is False
    assert out["level3_score"] == 0.3
    assert len(out["anomalies"]) == 1
    mock_judge.assert_called_once()
