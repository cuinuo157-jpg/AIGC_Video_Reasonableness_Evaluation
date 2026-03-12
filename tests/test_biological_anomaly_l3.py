"""测试 Level 3 MLLM 兜底和 ROI 工具。"""

import numpy as np
from unittest.mock import MagicMock

from src.biological_anomaly.roi_utils import crop_hand_roi, crop_mouth_roi
from src.biological_anomaly.mllm_bio_judge import judge_biological_anomaly_mllm


# ---------- ROI 裁剪工具 ----------


def test_crop_hand_roi():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:200, 200:350] = 255  # 手部区域
    hand_lm = np.zeros((21, 3), dtype=np.float32)
    # 归一化坐标：x ∈ [0.3, 0.5], y ∈ [0.2, 0.4]
    hand_lm[0] = [0.3, 0.2, 0.0]
    hand_lm[4] = [0.5, 0.4, 0.0]
    for i in range(1, 21):
        if i not in (0, 4):
            hand_lm[i] = [0.4, 0.3, 0.0]

    crop = crop_hand_roi(frame, hand_lm, padding=0.2)
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert crop.shape[2] == 3


def test_crop_mouth_roi():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    face_lm = np.random.rand(468, 3).astype(np.float32)
    # 将嘴部 landmark 集中到一个区域
    from src.biological_anomaly.anomaly_rules import MOUTH_CONSTRAINTS

    for idx in MOUTH_CONSTRAINTS["inner_lip_indices"]:
        face_lm[idx] = [0.5, 0.7, 0.0]

    crop = crop_mouth_roi(frame, face_lm, padding=0.3)
    assert crop.shape[0] > 0 and crop.shape[1] > 0


# ---------- MLLM 判定 ----------


def test_mllm_judge_skipped_no_client():
    result = judge_biological_anomaly_mllm(
        frames=[np.zeros((100, 100, 3), dtype=np.uint8)],
        keypoints_seq=[{"left_hand": None, "right_hand": None, "face": None}],
        suspicious_frames=[{"frame_idx": 0, "type": "hand_appear"}],
        mllm_client=None,
    )
    assert result["skipped"] is True


def test_mllm_judge_skipped_empty_suspicious():
    mock_mllm = MagicMock()
    result = judge_biological_anomaly_mllm(
        frames=[np.zeros((100, 100, 3), dtype=np.uint8)],
        keypoints_seq=[{"left_hand": None, "right_hand": None, "face": None}],
        suspicious_frames=[],
        mllm_client=mock_mllm,
    )
    assert result["skipped"] is True
    mock_mllm.judge_video_clip.assert_not_called()


def test_mllm_judge_returns_anomalies():
    """Mock MLLM 返回异常 -> 应正确传递。"""
    mock_mllm = MagicMock()
    mock_mllm.judge_video_clip.return_value = {
        "has_anomalies": True,
        "anomalies": [
            {"type": "finger_fusion", "description": "食指与中指融合", "severity": "severe"}
        ],
    }

    hand_lm = np.random.rand(21, 3).astype(np.float32)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = judge_biological_anomaly_mllm(
        frames=[frame],
        keypoints_seq=[{"left_hand": hand_lm, "right_hand": None, "face": None}],
        suspicious_frames=[{"frame_idx": 0, "type": "hand_velocity_jump", "hand": "left_hand"}],
        mllm_client=mock_mllm,
    )
    assert result["skipped"] is False
    assert result["has_anomalies"] is True
    assert len(result["anomalies"]) == 1


def test_mllm_judge_exception_handled():
    """MLLM 调用异常 -> 应返回 skipped。"""
    mock_mllm = MagicMock()
    mock_mllm.judge_video_clip.side_effect = RuntimeError("API error")

    hand_lm = np.random.rand(21, 3).astype(np.float32)
    result = judge_biological_anomaly_mllm(
        frames=[np.zeros((100, 100, 3), dtype=np.uint8)],
        keypoints_seq=[{"left_hand": hand_lm, "right_hand": None, "face": None}],
        suspicious_frames=[{"frame_idx": 0, "type": "hand_jitter", "hand": "left_hand"}],
        mllm_client=mock_mllm,
    )
    assert result["skipped"] is True
