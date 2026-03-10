from src.expression_naturalness.au_rules import check_au_combination
from src.expression_naturalness.temporal_analysis import compute_au_smoothness


def test_conflict_detection():
    aus = {"AU23": 2.0, "AU26": 3.0}
    violations = check_au_combination(aus)
    assert len(violations) > 0


def test_no_conflict():
    aus = {"AU06": 2.0, "AU12": 3.0}
    violations = check_au_combination(aus)
    assert len(violations) == 0


def test_au_smoothness_stable():
    sequence = [1.0, 1.1, 1.0, 0.9, 1.0, 1.1]
    score = compute_au_smoothness(sequence)
    assert score > 0.8


def test_au_smoothness_jumpy():
    sequence = [0.0, 4.0, 0.0, 4.0, 0.0, 4.0]
    score = compute_au_smoothness(sequence)
    assert score < 0.5
