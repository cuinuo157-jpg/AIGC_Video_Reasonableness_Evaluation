import numpy as np
from unittest.mock import MagicMock

from src.motion_logic.analyzer import MotionLogicAnalyzer
from src.motion_logic.config import MotionLogicConfig


def test_motion_logic_with_flows():
    hub = MagicMock()
    flows = [
        (np.ones((50, 50)) * 0.5, np.ones((50, 50)) * 0.5) for _ in range(10)
    ]
    hub.get.side_effect = lambda k: {"optical_flow": flows}.get(k)
    result = MotionLogicAnalyzer(
        MotionLogicConfig(enable_mllm=False)
    ).analyze(hub)
    assert result.applicable is True
    assert 0.0 <= result.motion_logic_score <= 1.0


def test_motion_logic_no_motion():
    hub = MagicMock()
    hub.get.side_effect = lambda k: {"optical_flow": []}.get(k)
    result = MotionLogicAnalyzer(
        MotionLogicConfig(enable_mllm=False)
    ).analyze(hub)
    assert result.applicable is False
