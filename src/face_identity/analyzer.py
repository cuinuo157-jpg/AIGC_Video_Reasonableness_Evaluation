from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import FaceIdentityConfig
from .csim_scorer import CSIMScorer, CSIMResult, DropEvent
from .face_tracker import FaceTracker, FaceTrack


@dataclass
class FaceIdentityResult:
    applicable: bool
    skip_reason: str | None = None
    face_tracks: list[FaceTrack] = field(default_factory=list)
    csim_ref: float = 0.0
    csim_adj: float = 0.0
    csim_min: float = 0.0
    drop_events: list[DropEvent] = field(default_factory=list)
    identity_score: float = 0.0


class FaceIdentityAnalyzer:
    def __init__(self, config: FaceIdentityConfig | None = None) -> None:
        self.config = config or FaceIdentityConfig()
        self._tracker = FaceTracker()
        self._scorer = CSIMScorer(self.config)

    def analyze(self, hub: Any) -> FaceIdentityResult:
        frame_data = hub.get("face_embedding")
        total_faces = sum(fd["num_faces"] for fd in frame_data)

        if total_faces == 0:
            return FaceIdentityResult(applicable=False, skip_reason="no face detected")

        tracks = self._tracker.track(frame_data)
        if not tracks:
            return FaceIdentityResult(applicable=False, skip_reason="no face tracks")

        main_track = max(tracks, key=lambda t: len(t.embeddings))
        csim = self._scorer.compute(main_track.embeddings)

        return FaceIdentityResult(
            applicable=True,
            face_tracks=tracks,
            csim_ref=csim.csim_ref,
            csim_adj=csim.csim_adj,
            csim_min=csim.csim_min,
            drop_events=csim.drop_events,
            identity_score=csim.identity_score,
        )
