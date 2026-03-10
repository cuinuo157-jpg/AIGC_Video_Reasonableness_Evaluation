import numpy as np

from src.physics_consistency.pixel_drift import detect_pixel_drift


def test_no_drift_in_static():
    flows = [(np.zeros((50, 50)), np.zeros((50, 50))) for _ in range(10)]
    mask = np.ones((50, 50), dtype=bool)
    assert len(detect_pixel_drift(flows, static_mask=mask)) == 0


def test_detect_drift():
    flows = [
        (np.ones((50, 50)) * 2.0, np.zeros((50, 50))) for _ in range(10)
    ]
    mask = np.ones((50, 50), dtype=bool)
    assert len(detect_pixel_drift(flows, static_mask=mask)) > 0
