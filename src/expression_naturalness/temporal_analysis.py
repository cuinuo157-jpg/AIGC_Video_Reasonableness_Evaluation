from __future__ import annotations

import numpy as np


def compute_au_smoothness(sequence: list[float], window: int = 3) -> float:
    if len(sequence) < 2:
        return 1.0
    arr = np.array(sequence)
    diffs = np.abs(np.diff(arr))
    max_possible_diff = 5.0
    normalized_diffs = diffs / max_possible_diff
    smoothness = 1.0 - float(np.mean(normalized_diffs))
    return float(np.clip(smoothness, 0, 1))


def compute_all_au_smoothness(
    au_sequences: dict[str, list[float]],
) -> dict[str, float]:
    return {
        au: compute_au_smoothness(seq)
        for au, seq in au_sequences.items()
        if len(seq) >= 2
    }
