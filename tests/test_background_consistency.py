import numpy as np

from src.background_consistency.static_region_analysis import compute_residual_score
from src.background_consistency.depth_consistency import compute_depth_consistency


def test_residual_score_identical():
    frames = [np.ones((50, 50, 3), dtype=np.uint8) * 128] * 5
    assert compute_residual_score(frames) > 0.95


def test_residual_score_different():
    frames = [
        np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        for _ in range(5)
    ]
    assert compute_residual_score(frames) < 0.8


def test_depth_consistency_stable():
    # Shared spatial pattern with small per-frame perturbation
    base = np.random.rand(50, 50) * 10.0
    depths = [base + np.random.randn(50, 50) * 0.01 for _ in range(5)]
    assert compute_depth_consistency(depths) > 0.9


def test_depth_consistency_flickering():
    depths = [
        np.ones((50, 50)) * (5.0 if i % 2 == 0 else 0.5) for i in range(10)
    ]
    assert compute_depth_consistency(depths) < 0.5
