from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import BiologicalAnomalyConfig
from .eye_anomaly import detect_eye_anomalies
from .hand_anomaly import detect_hand_anomalies


@dataclass
class BiologicalAnomalyResult:
    applicable: bool
    skip_reason: str | None = None
    eye_anomalies: list[dict] = field(default_factory=list)
    hand_anomalies: list[dict] = field(default_factory=list)
    mouth_anomalies: list[dict] = field(default_factory=list)
    anomaly_count: int = 0
    bio_quality_score: float = 1.0


class BiologicalAnomalyAnalyzer:
    def __init__(self, config: BiologicalAnomalyConfig | None = None) -> None:
        self.config = config or BiologicalAnomalyConfig()

    def analyze(self, hub: Any) -> BiologicalAnomalyResult:
        face_data = hub.get("face_embedding")
        has_faces = any(fd["num_faces"] > 0 for fd in face_data)
        if not has_faces:
            return BiologicalAnomalyResult(
                applicable=False, skip_reason="no face or hand detected"
            )

        eye_anomalies: list[dict] = []
        hand_anomalies: list[dict] = []
        mouth_anomalies: list[dict] = []

        try:
            keypoints = hub.get("keypoints")
            ear_seq = [kp.get("ear", 0.35) for kp in keypoints]
            eye_anomalies = detect_eye_anomalies(ear_seq, fps=30.0)
            finger_counts = [kp.get("finger_count", 5) for kp in keypoints]
            hand_anomalies = detect_hand_anomalies(finger_counts=finger_counts)
        except (KeyError, TypeError):
            pass

        total = len(eye_anomalies) + len(hand_anomalies) + len(mouth_anomalies)
        n_frames = len(face_data)
        score = float(np.clip(1.0 - total / max(n_frames, 1), 0, 1))

        return BiologicalAnomalyResult(
            applicable=True,
            eye_anomalies=eye_anomalies,
            hand_anomalies=hand_anomalies,
            mouth_anomalies=mouth_anomalies,
            anomaly_count=total,
            bio_quality_score=score,
        )
