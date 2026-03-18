import numpy as np
from unittest.mock import MagicMock

from src.motion_logic.analyzer import MotionLogicAnalyzer
from src.motion_logic.config import MotionLogicConfig


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
