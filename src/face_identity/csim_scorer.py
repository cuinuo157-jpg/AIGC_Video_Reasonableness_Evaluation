from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import FaceIdentityConfig


@dataclass
class DropEvent:
    frame_idx: int
    similarity_before: float
    similarity_after: float
    drop_magnitude: float


@dataclass
class CSIMResult:
    csim_ref: float
    csim_adj: float
    csim_min: float
    drop_events: list[DropEvent]
    identity_score: float


class CSIMScorer:
    def __init__(self, config: FaceIdentityConfig | None = None) -> None:
        self.config = config or FaceIdentityConfig()

    def compute(self, embeddings: list[np.ndarray]) -> CSIMResult:
        if len(embeddings) < 2:
            return CSIMResult(1.0, 1.0, 1.0, [], 1.0)

        ref = embeddings[0]
        ref_sims = [float(np.dot(ref, e)) for e in embeddings[1:]]
        csim_ref = float(np.mean(ref_sims)) if ref_sims else 1.0

        adj_sims = [
            float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ]
        csim_adj = float(np.mean(adj_sims))

        all_sims = ref_sims + adj_sims
        csim_min = float(np.min(all_sims)) if all_sims else 1.0

        drop_events = self._detect_drops(adj_sims)

        c = self.config
        drop_penalty = len(drop_events) / max(len(embeddings) - 1, 1)
        identity_score = (
            c.csim_ref_weight * max(csim_ref, 0)
            + c.csim_adj_weight * max(csim_adj, 0)
            + c.csim_min_weight * max(csim_min, 0)
            - c.drop_penalty_weight * drop_penalty
        )
        identity_score = float(np.clip(identity_score, 0, 1))

        return CSIMResult(csim_ref, csim_adj, csim_min, drop_events, identity_score)

    def _detect_drops(self, adj_sims: list[float]) -> list[DropEvent]:
        drops = []
        w = self.config.drop_window
        for i in range(len(adj_sims)):
            start = max(0, i - w)
            window_mean = float(np.mean(adj_sims[start:i])) if i > 0 else 1.0
            drop_mag = window_mean - adj_sims[i]
            if drop_mag > self.config.drop_threshold:
                drops.append(
                    DropEvent(
                        frame_idx=i + 1,
                        similarity_before=window_mean,
                        similarity_after=adj_sims[i],
                        drop_magnitude=drop_mag,
                    )
                )
        return drops
