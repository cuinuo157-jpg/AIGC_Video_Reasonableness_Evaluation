import numpy as np

from src.motion_logic.smoothness_scorer import compute_flow_acceleration_smoothness


def test_smooth_flow():
    flows = [
        (np.ones((50, 50)) * i * 0.1, np.ones((50, 50)) * i * 0.1)
        for i in range(10)
    ]
    assert compute_flow_acceleration_smoothness(flows) > 0.8


def test_jumpy_flow():
    flows = [
        (
            np.ones((50, 50)) * (10.0 if i % 2 == 0 else 0.0),
            np.ones((50, 50)) * (10.0 if i % 2 == 0 else 0.0),
        )
        for i in range(10)
    ]
    assert compute_flow_acceleration_smoothness(flows) < 0.5
