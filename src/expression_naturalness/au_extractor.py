from __future__ import annotations

import numpy as np


class AUExtractor:
    def __init__(self) -> None:
        self._detector = None

    def _ensure_detector(self) -> None:
        if self._detector is None:
            from feat import Detector

            self._detector = Detector(au_model="xgb")

    def extract(self, frame: np.ndarray) -> dict[str, float]:
        self._ensure_detector()
        result = self._detector.detect_image(frame)
        if result.aus is None or len(result.aus) == 0:
            return {}
        au_values = result.aus.values[0]
        au_names = list(result.aus.columns)
        return {name: float(val) for name, val in zip(au_names, au_values)}

    def extract_sequence(
        self, frames: list[np.ndarray]
    ) -> list[dict[str, float]]:
        return [self.extract(f) for f in frames]
