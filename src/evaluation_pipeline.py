from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Iterable

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
from src.feature_hub.hub import FeatureHub, VideoProcessingConfig, create_default_hub

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

DEFAULT_ANOMALY_TYPES = (
    "face_identity",
    "expression",
    "biological_anomaly",
    "motion_logic",
    "physics",
)

DEFAULT_TOP5_TYPES = (
    "face_identity",
    "biological_anomaly",
    "motion_logic",
    "physics",
    "temporal_coherence",
)

_ANOMALY_TYPE_ALIASES = {
    "identity": "face_identity",
    "face": "face_identity",
    "face_identity": "face_identity",
    "expression": "expression",
    "bio": "biological_anomaly",
    "biological": "biological_anomaly",
    "biological_anomaly": "biological_anomaly",
    "motion": "motion_logic",
    "motion_logic": "motion_logic",
    "physics": "physics",
    "身份": "face_identity",
    "表情": "expression",
    "生物异常": "biological_anomaly",
    "运动": "motion_logic",
    "物理": "physics",
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
        video_config: VideoProcessingConfig | None = None,
        parallel: bool = False,
        max_workers: int | None = None,
        au_backend: str | None = None,
        au_external_python: str | None = None,
    ) -> None:
        self.device = device
        self.weights = weights or DEFAULT_WEIGHTS
        self.enable_mllm = enable_mllm
        self._mllm_client = mllm_client
        self.video_config = video_config or VideoProcessingConfig()
        self.parallel = parallel
        self.max_workers = max_workers
        self.au_backend = au_backend
        self.au_external_python = au_external_python
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
        return create_default_hub(
            video_path,
            self.device,
            video_config=self.video_config,
            runtime_options={
                "au_backend": self.au_backend,
                "au_external_python": self.au_external_python,
            },
        )

    def _normalize_dimensions(
        self,
        selected_dimensions: str | Iterable[str] | None,
    ) -> list[str]:
        if selected_dimensions is None:
            return list(self._analyzers.keys())

        if isinstance(selected_dimensions, str):
            candidates = [
                item.strip() for item in selected_dimensions.split(",") if item.strip()
            ]
        else:
            candidates = [str(item).strip() for item in selected_dimensions if str(item).strip()]

        normalized: list[str] = []
        unknown: list[str] = []
        for name in candidates:
            if name in self._analyzers:
                target = name
            else:
                target = _ANOMALY_TYPE_ALIASES.get(name)
            if target is None or target not in self._analyzers:
                unknown.append(name)
                continue
            if target not in normalized:
                normalized.append(target)

        if unknown:
            raise ValueError(
                f"Unknown dimensions: {unknown}. "
                f"Available: {list(self._analyzers.keys())}"
            )
        if not normalized:
            raise ValueError("No dimensions selected")
        return normalized

    def _run_analyzer(
        self,
        name: str,
        analyzer: Any,
        hub: FeatureHub,
    ) -> DimensionResult:
        try:
            raw = analyzer.analyze(hub)
            score_attr = _SCORE_ATTR_MAP[name]
            score = getattr(raw, score_attr, None) if raw.applicable else None
            return DimensionResult(
                applicable=raw.applicable,
                skip_reason=getattr(raw, "skip_reason", None),
                score=score,
                weight=self.weights.get(name, 0.0),
                details=raw,
            )
        except Exception as exc:
            return DimensionResult(
                applicable=False,
                skip_reason=f"error: {exc}",
                weight=self.weights.get(name, 0.0),
            )

    def evaluate(
        self,
        video_path: str,
        selected_dimensions: str | Iterable[str] | None = None,
        parallel: bool | None = None,
        max_workers: int | None = None,
    ) -> EvaluationReport:
        """对视频执行评测，支持按维度选择与并发调度。"""
        report, _hub = self.evaluate_with_hub(
            video_path,
            selected_dimensions=selected_dimensions,
            parallel=parallel,
            max_workers=max_workers,
        )
        return report

    def evaluate_with_hub(
        self,
        video_path: str,
        selected_dimensions: str | Iterable[str] | None = None,
        parallel: bool | None = None,
        max_workers: int | None = None,
    ) -> tuple[EvaluationReport, FeatureHub]:
        """对视频执行评测，并返回复用的 FeatureHub。"""
        hub = self._create_hub(video_path)
        dimension_names = self._normalize_dimensions(selected_dimensions)
        results: dict[str, DimensionResult] = {}
        use_parallel = self.parallel if parallel is None else parallel
        worker_count = max_workers or self.max_workers or min(4, len(dimension_names))

        if use_parallel and len(dimension_names) > 1 and worker_count > 1:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(self._run_analyzer, name, self._analyzers[name], hub): name
                    for name in dimension_names
                }
                for future in as_completed(future_map):
                    name = future_map[future]
                    results[name] = future.result()
        else:
            for name in dimension_names:
                results[name] = self._run_analyzer(name, self._analyzers[name], hub)

        active, final_score = _redistribute_weights(results)

        report = EvaluationReport(
            dimensions=results,
            active_dimensions=list(active.keys()),
            final_score=final_score,
        )
        return report, hub

    def detect_anomalies(
        self,
        video_path: str,
        anomaly_types: str | Iterable[str] | None = None,
        parallel: bool | None = None,
        max_workers: int | None = None,
    ) -> EvaluationReport:
        """统一的五类异常检测接口。"""
        selected = anomaly_types or DEFAULT_ANOMALY_TYPES
        return self.evaluate(
            video_path,
            selected_dimensions=selected,
            parallel=parallel,
            max_workers=max_workers,
        )

    def detect_anomalies_with_hub(
        self,
        video_path: str,
        anomaly_types: str | Iterable[str] | None = None,
        parallel: bool | None = None,
        max_workers: int | None = None,
    ) -> tuple[EvaluationReport, FeatureHub]:
        """统一的五类异常检测接口，并返回复用的 FeatureHub。"""
        selected = anomaly_types or DEFAULT_ANOMALY_TYPES
        return self.evaluate_with_hub(
            video_path,
            selected_dimensions=selected,
            parallel=parallel,
            max_workers=max_workers,
        )
