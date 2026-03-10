from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .au_extractor import AUExtractor
from .au_rules import Violation, check_au_combination
from .config import ExpressionConfig
from .temporal_analysis import compute_all_au_smoothness


@dataclass
class ExpressionResult:
    applicable: bool
    skip_reason: str | None = None
    au_sequences: dict[str, list[float]] = field(default_factory=dict)
    combination_violations: list[Violation] = field(default_factory=list)
    temporal_smoothness: float = 0.0
    expression_score: float = 0.0


class ExpressionAnalyzer:
    def __init__(self, config: ExpressionConfig | None = None) -> None:
        self.config = config or ExpressionConfig()
        self._extractor = AUExtractor()

    def analyze(self, hub: Any) -> ExpressionResult:
        face_data = hub.get("face_embedding")
        has_faces = any(fd["num_faces"] > 0 for fd in face_data)
        if not has_faces:
            return ExpressionResult(
                applicable=False, skip_reason="no face detected"
            )

        try:
            frames = hub.get("video_frames")
        except KeyError:
            return ExpressionResult(
                applicable=False, skip_reason="video_frames not available"
            )

        au_per_frame = self._extractor.extract_sequence(frames)
        if not any(au_per_frame):
            return ExpressionResult(
                applicable=False, skip_reason="no AU detected"
            )

        all_aus: set[str] = set()
        for aus in au_per_frame:
            all_aus.update(aus.keys())

        au_sequences = {
            au: [f.get(au, 0.0) for f in au_per_frame] for au in all_aus
        }

        violations: list[Violation] = []
        for aus in au_per_frame:
            violations.extend(check_au_combination(aus))

        smoothness_scores = compute_all_au_smoothness(au_sequences)
        temporal_smoothness = (
            float(np.mean(list(smoothness_scores.values())))
            if smoothness_scores
            else 1.0
        )

        c = self.config
        violation_penalty = min(
            len(violations) / max(len(au_per_frame), 1), 1.0
        )
        expression_score = (
            c.au_combination_weight * (1.0 - violation_penalty)
            + c.au_smoothness_weight * temporal_smoothness
            + c.flow_consistency_weight * 1.0
        )
        expression_score = float(np.clip(expression_score, 0, 1))

        return ExpressionResult(
            applicable=True,
            au_sequences=au_sequences,
            combination_violations=violations,
            temporal_smoothness=temporal_smoothness,
            expression_score=expression_score,
        )
