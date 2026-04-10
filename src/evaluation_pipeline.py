from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.face_identity.analyzer import FaceIdentityAnalyzer
from src.expression_naturalness.analyzer import ExpressionAnalyzer
from src.biological_anomaly.analyzer import BiologicalAnomalyAnalyzer
from src.biological_anomaly.config import BiologicalAnomalyConfig
from src.motion_logic.analyzer import MotionLogicAnalyzer
from src.motion_logic.config import MotionLogicConfig
from src.physics_consistency.analyzer import PhysicsConsistencyAnalyzer
from src.physics_consistency.config import PhysicsConfig
from src.background_consistency.analyzer import BackgroundConsistencyAnalyzer
from src.perceptual_quality.analyzer import PerceptualQualityAnalyzer
from src.temporal_coherence.analyzer import TemporalCoherenceAnalyzer
from src.feature_hub.hub import create_default_hub, FeatureHub

DEFAULT_WEIGHTS = {
    "face_identity": 0.12,
    "expression": 0.12,
    "biological_anomaly": 0.15,
    "motion_logic": 0.12,
    "temporal_coherence": 0.10,
    "physics": 0.12,
    "background": 0.15,
    "perceptual_quality": 0.12,
}


@dataclass
class DimensionResult:
    applicable: bool = True
    skip_reason: str | None = None
    score: float | None = None
    weight: float = 0.0
    details: Any = None


@dataclass
class EvaluationReport:
    dimensions: dict[str, DimensionResult] = field(default_factory=dict)
    active_dimensions: list[str] = field(default_factory=list)
    final_score: float = 0.0
    video_type: str = "unknown"


def _redistribute_weights(
    results: dict[str, DimensionResult],
) -> tuple[dict[str, DimensionResult], float]:
    """对适用维度进行权重归一化并计算加权总分。"""
    active = {k: v for k, v in results.items() if v.applicable and v.score is not None}
    if not active:
        return {}, 0.0
    total_w = sum(v.weight for v in active.values())
    if total_w > 0:
        for v in active.values():
            v.weight = v.weight / total_w
    final = sum(v.weight * v.score for v in active.values())
    return active, float(np.clip(final, 0, 1))


_SCORE_ATTR_MAP = {
    "face_identity": "identity_score",
    "expression": "expression_score",
    "biological_anomaly": "bio_quality_score",
    "motion_logic": "motion_logic_score",
    "temporal_coherence": "temporal_coherence_score",
    "physics": "physics_score",
    "background": "background_score",
    "perceptual_quality": "perceptual_quality_score",
}


class EvaluationPipeline:
    """七维度统一评测流水线。"""

    def __init__(
        self,
        device: str = "cuda",
        weights: dict[str, float] | None = None,
        enable_mllm: bool = False,
        mllm_client: Any = None,
    ) -> None:
        self.device = device
        self.weights = weights or DEFAULT_WEIGHTS
        self.enable_mllm = enable_mllm
        self._mllm_client = mllm_client
        mllm_for_analyzers = mllm_client if enable_mllm else None
        self._analyzers: dict[str, Any] = {
            "face_identity": FaceIdentityAnalyzer(),
            "expression": ExpressionAnalyzer(),
            "biological_anomaly": BiologicalAnomalyAnalyzer(
                config=BiologicalAnomalyConfig(enable_mllm=enable_mllm),
                mllm_client=mllm_for_analyzers,
            ),
            "motion_logic": MotionLogicAnalyzer(
                config=MotionLogicConfig(enable_mllm=enable_mllm),
                mllm_client=mllm_for_analyzers,
            ),
            "temporal_coherence": TemporalCoherenceAnalyzer(),
            "physics": PhysicsConsistencyAnalyzer(
                config=PhysicsConfig(enable_mllm=enable_mllm),
                mllm_client=mllm_for_analyzers,
            ),
            "background": BackgroundConsistencyAnalyzer(),
            "perceptual_quality": PerceptualQualityAnalyzer(),
        }

    def _create_hub(self, video_path: str) -> FeatureHub:
        return create_default_hub(video_path, self.device)

    def evaluate(self, video_path: str) -> EvaluationReport:
        """对视频执行七维度评测，返回结构化报告。"""
        hub = self._create_hub(video_path)
        results: dict[str, DimensionResult] = {}

        for name, analyzer in self._analyzers.items():
            try:
                raw = analyzer.analyze(hub)
                score_attr = _SCORE_ATTR_MAP[name]
                score = getattr(raw, score_attr, None) if raw.applicable else None
                results[name] = DimensionResult(
                    applicable=raw.applicable,
                    skip_reason=getattr(raw, "skip_reason", None),
                    score=score,
                    weight=self.weights.get(name, 0.0),
                    details=raw,
                )
            except Exception as e:
                results[name] = DimensionResult(
                    applicable=False,
                    skip_reason=f"error: {e}",
                    weight=self.weights.get(name, 0.0),
                )

        active, final_score = _redistribute_weights(results)

        return EvaluationReport(
            dimensions=results,
            active_dimensions=list(active.keys()),
            final_score=final_score,
        )
