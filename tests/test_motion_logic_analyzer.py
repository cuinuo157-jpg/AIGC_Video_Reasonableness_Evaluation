import numpy as np
from unittest.mock import MagicMock

from src.motion_logic.analyzer import MotionLogicAnalyzer
from src.motion_logic.config import MotionLogicConfig
from src.feature_hub.extractors.subject_segmentation import SubjectSegmentationResult


def _make_hub(features: dict) -> MagicMock:
    """创建模拟 FeatureHub，支持 has_extractor / get。"""
    hub = MagicMock()
    hub.has_extractor.side_effect = lambda k: k in features
    hub.get.side_effect = lambda k: features[k]
    return hub


def test_motion_logic_with_flows():
    flows = [
        (np.ones((50, 50)) * 0.5, np.ones((50, 50)) * 0.5) for _ in range(10)
    ]
    hub = _make_hub({"optical_flow": flows})
    result = MotionLogicAnalyzer(
        MotionLogicConfig(enable_mllm=False)
    ).analyze(hub)
    assert result.applicable is True
    assert 0.0 <= result.motion_logic_score <= 1.0


def test_motion_logic_with_raft_flow():
    """当 raft_flow 可用时优先使用。"""
    flows = [
        (np.ones((50, 50)) * 2.0, np.ones((50, 50)) * 2.0) for _ in range(10)
    ]
    hub = _make_hub({"raft_flow": flows, "optical_flow": []})
    result = MotionLogicAnalyzer(
        MotionLogicConfig(enable_mllm=False)
    ).analyze(hub)
    assert result.applicable is True
    assert result.dynamics_score > 0


def test_motion_logic_no_motion():
    hub = _make_hub({"optical_flow": []})
    result = MotionLogicAnalyzer(
        MotionLogicConfig(enable_mllm=False)
    ).analyze(hub)
    assert result.applicable is False


def test_motion_logic_with_subject_masks():
    """集成测试: 带主体 mask 的动态度评分。"""
    h, w, n = 50, 50, 10
    # 主体区域大运动，背景静止
    masks = []
    flows = []
    for _ in range(n):
        m = np.zeros((h, w), dtype=bool)
        m[:25, :25] = True
        masks.append(m)

        fx = np.zeros((h, w), dtype=np.float32)
        fy = np.zeros((h, w), dtype=np.float32)
        fx[:25, :25] = 8.0
        fy[:25, :25] = 8.0
        flows.append((fx, fy))

    seg_result = SubjectSegmentationResult(
        masks=masks,
        subject_ratios=[0.25] * n,
        method="sam2_grounding",
    )

    hub = _make_hub({
        "optical_flow": flows,
        "subject_masks": seg_result,
    })
    result = MotionLogicAnalyzer(
        MotionLogicConfig(enable_mllm=False)
    ).analyze(hub)

    assert result.applicable is True
    assert result.subject_motion_detail is not None
    assert result.subject_motion_detail.perceptual_score > 0
    assert result.dynamics_detail.subject_perceptual is not None


def test_motion_logic_subject_masks_none_method():
    """subject_masks method='none' 时应退回全局模式。"""
    flows = [
        (np.ones((50, 50)) * 2.0, np.ones((50, 50)) * 2.0) for _ in range(10)
    ]
    seg_result = SubjectSegmentationResult(
        masks=[],
        subject_ratios=[],
        method="none",
    )
    hub = _make_hub({
        "optical_flow": flows,
        "subject_masks": seg_result,
    })
    result = MotionLogicAnalyzer(
        MotionLogicConfig(enable_mllm=False)
    ).analyze(hub)

    assert result.applicable is True
    assert result.subject_motion_detail is None
    assert result.dynamics_detail.subject_perceptual is None
