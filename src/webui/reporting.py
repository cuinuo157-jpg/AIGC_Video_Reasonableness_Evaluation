from __future__ import annotations

from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation_pipeline import DimensionResult, EvaluationReport
from .service import DIMENSION_CATALOG, WebUIRunConfig


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except Exception:
        return None


def _metric(label: str, value: Any, kind: str = "number") -> dict[str, Any]:
    return {"label": label, "value": value, "kind": kind}


def _event(title: str, detail: str, severity: str = "info") -> dict[str, str]:
    return {"title": title, "detail": detail, "severity": severity}


def _top_events(items: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    return items[:limit]


def _score_band(score: float | None) -> str:
    if score is None:
        return "na"
    if score >= 0.85:
        return "excellent"
    if score >= 0.7:
        return "good"
    if score >= 0.5:
        return "warning"
    return "critical"


def _summarize_face_identity(raw: Any) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    metrics = [
        _metric("身份分", _float(getattr(raw, "identity_score", None)), "score"),
        _metric("参考相似度", _float(getattr(raw, "csim_ref", None)), "score"),
        _metric("相邻相似度", _float(getattr(raw, "csim_adj", None)), "score"),
        _metric("最低相似度", _float(getattr(raw, "csim_min", None)), "score"),
        _metric("轨迹数", len(getattr(raw, "face_tracks", []) or []), "count"),
        _metric("突降事件", len(getattr(raw, "drop_events", []) or []), "count"),
    ]
    drop_events = [
        _event(
            f"帧 {ev.frame_idx} 身份相似度突降",
            f"{ev.similarity_before:.3f} -> {ev.similarity_after:.3f}，跌幅 {ev.drop_magnitude:.3f}",
            "warning",
        )
        for ev in getattr(raw, "drop_events", []) or []
    ]
    highlights = [
        f"最低身份相似度 {getattr(raw, 'csim_min', 0.0):.3f}",
        f"主人脸轨迹数 {len(getattr(raw, 'face_tracks', []) or [])}",
    ]
    return metrics, highlights, _top_events(drop_events)


def _summarize_expression(raw: Any) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    violations = getattr(raw, "combination_violations", []) or []
    metrics = [
        _metric("表情分", _float(getattr(raw, "expression_score", None)), "score"),
        _metric("时序平滑度", _float(getattr(raw, "temporal_smoothness", None)), "score"),
        _metric("AU 规则冲突", len(violations), "count"),
        _metric("AU 通道数", len(getattr(raw, "au_sequences", {}) or {}), "count"),
    ]
    events = []
    for violation in violations[:5]:
        rule = getattr(violation, "rule_name", "规则冲突")
        reason = getattr(violation, "description", "") or getattr(violation, "message", "")
        events.append(_event(rule, reason, "warning"))
    highlights = [
        f"检测到 {len(violations)} 个 AU 组合异常",
        f"表情过渡平滑度 {getattr(raw, 'temporal_smoothness', 0.0):.3f}",
    ]
    return metrics, highlights, events


def _summarize_biological(raw: Any) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    body_scores = getattr(raw, "body_part_scores", {}) or {}
    metrics = [
        _metric("综合分", _float(getattr(raw, "bio_quality_score", None)), "score"),
        _metric("Level 1", _float(getattr(raw, "level1_score", None)), "score"),
        _metric("Level 2", _float(getattr(raw, "level2_score", None)), "score"),
        _metric("Level 3", _float(getattr(raw, "level3_score", None)), "score"),
        _metric("总异常数", getattr(raw, "anomaly_count", 0), "count"),
        _metric("眼部异常", len(getattr(raw, "eye_anomalies", []) or []), "count"),
        _metric("嘴部异常", len(getattr(raw, "mouth_anomalies", []) or []), "count"),
        _metric("手部异常", len(getattr(raw, "hand_anomalies", []) or []), "count"),
        _metric("身体异常", len(getattr(raw, "body_anomalies", []) or []), "count"),
    ]
    all_events = []
    for section_name, anomalies in (
        ("眼部", getattr(raw, "eye_anomalies", []) or []),
        ("嘴部", getattr(raw, "mouth_anomalies", []) or []),
        ("手部", getattr(raw, "hand_anomalies", []) or []),
        ("身体", getattr(raw, "body_anomalies", []) or []),
        ("MLLM", getattr(raw, "mllm_anomalies", []) or []),
    ):
        for anomaly in anomalies:
            detail = anomaly.get("type") or anomaly.get("description") or anomaly.get("reason") or "异常"
            frame_idx = anomaly.get("frame_idx", "-")
            severity = anomaly.get("severity", "warning")
            all_events.append(_event(f"{section_name} / 帧 {frame_idx}", str(detail), str(severity)))
    body_hint = ", ".join(
        f"{name}:{score:.2f}" for name, score in list(body_scores.items())[:4]
    ) or "暂无部位稳定度统计"
    highlights = [
        f"共定位 {getattr(raw, 'anomaly_count', 0)} 处生物特征异常",
        f"部位稳定度 {body_hint}",
    ]
    return metrics, highlights, _top_events(all_events)


def _summarize_motion(raw: Any) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    naturalness_issues = getattr(raw, "naturalness_issues", []) or []
    traj_detail = getattr(raw, "trajectory_curvature_detail", None)
    subject_detail = getattr(raw, "subject_motion_detail", None)
    metrics = [
        _metric("综合分", _float(getattr(raw, "motion_logic_score", None)), "score"),
        _metric("动态度", _float(getattr(raw, "dynamics_score", None)), "score"),
        _metric("平滑度", _float(getattr(raw, "smoothness_score", None)), "score"),
        _metric("运动自然度", _float(getattr(raw, "naturalness_score", None)), "score"),
        _metric("轨迹异常数", getattr(traj_detail, "abnormal_event_count", 0) if traj_detail else 0, "count"),
        _metric("主体感知运动分", _float(getattr(subject_detail, "perceptual_score", None) if subject_detail else None), "score"),
    ]
    events = [_event("自然度问题", issue, "warning") for issue in naturalness_issues[:5]]
    if traj_detail and getattr(traj_detail, "abnormal_ratio", None) is not None:
        events.append(
            _event(
                "轨迹曲率异常",
                f"异常占比 {traj_detail.abnormal_ratio:.3f}，事件数 {traj_detail.abnormal_event_count}",
                "warning",
            )
        )
    highlights = [
        f"动态度 {getattr(raw, 'dynamics_score', 0.0):.3f}，平滑度 {getattr(raw, 'smoothness_score', 0.0):.3f}",
        f"自然度问题 {len(naturalness_issues)} 条",
    ]
    return metrics, highlights, _top_events(events)


def _summarize_physics(raw: Any) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    violations = getattr(raw, "vlm_violations", []) or []
    drift_events = getattr(raw, "drift_events", []) or []
    metrics = [
        _metric("综合分", _float(getattr(raw, "physics_score", None)), "score"),
        _metric("像素漂移分", _float(getattr(raw, "drift_score", None)), "score"),
        _metric("VLM 物理分", _float(getattr(raw, "vlm_score", None)), "score"),
        _metric("漂移事件", len(drift_events), "count"),
        _metric("物理违规", len(violations), "count"),
    ]
    events = []
    for violation in violations[:5]:
        title = violation.get("type") or "物理违规"
        detail = violation.get("description") or violation.get("reason") or str(violation)
        events.append(_event(title, detail, "critical"))
    for drift in drift_events[:3]:
        detail = f"平均幅度 {drift.get('avg_magnitude', 0):.3f}, 持续 {drift.get('duration_frames', 0)} 帧"
        events.append(_event("像素漂移", detail, "warning"))
    highlights = [
        f"漂移事件 {len(drift_events)} 个，物理违规 {len(violations)} 个",
        str(getattr(raw, "vlm_reasoning", "")).strip()[:140] or "未返回额外物理推理文本",
    ]
    return metrics, highlights, _top_events(events)


def _summarize_background(raw: Any) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    metrics = [
        _metric("综合分", _float(getattr(raw, "background_score", None)), "score"),
        _metric("静态残差", _float(getattr(raw, "residual_score", None)), "score"),
        _metric("单应稳定性", _float(getattr(raw, "homography_stability", None)), "score"),
        _metric("深度一致性", _float(getattr(raw, "depth_consistency", None)), "score"),
    ]
    highlights = [
        f"背景残差 {getattr(raw, 'residual_score', 0.0):.3f}",
        f"单应稳定性 {getattr(raw, 'homography_stability', 0.0):.3f}",
    ]
    return metrics, highlights, []


def _summarize_temporal(raw: Any) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    abnormal_events = getattr(raw, "abnormal_events", []) or []
    temporal_events = getattr(raw, "temporal_events", []) or []
    metrics = [
        _metric("综合分", _float(getattr(raw, "temporal_coherence_score", None)), "score"),
        _metric("总事件数", len(temporal_events), "count"),
        _metric("异常事件", len(abnormal_events), "count"),
    ]
    events = [
        _event(
            f"{event.event_type} / 帧 {event.frame_idx}",
            f"track {event.track_id}，原因 {event.reason}",
            "warning",
        )
        for event in abnormal_events[:5]
    ]
    highlights = [
        f"抽样轨迹事件 {len(temporal_events)} 个",
        f"异常出现/消失事件 {len(abnormal_events)} 个",
    ]
    return metrics, highlights, events


def _summarize_perceptual(raw: Any) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    frame_scores = getattr(raw, "frame_quality_scores", []) or []
    weakest_idx = int(np.argmin(frame_scores)) if frame_scores else -1
    weakest_score = float(np.min(frame_scores)) if frame_scores else 0.0
    metrics = [
        _metric("综合分", _float(getattr(raw, "perceptual_quality_score", None)), "score"),
        _metric("清晰度", _float(getattr(raw, "blur_score", None)), "score"),
        _metric("一致性", _float(getattr(raw, "consistency_score", None)), "score"),
        _metric("瑕疵分", _float(getattr(raw, "artifact_score", None)), "score"),
        _metric("分析帧数", len(frame_scores), "count"),
    ]
    highlights = [
        f"最弱帧 {weakest_idx if weakest_idx >= 0 else '-'}，质量分 {weakest_score:.3f}",
        f"平均帧质量 {float(np.mean(frame_scores)):.3f}" if frame_scores else "未返回逐帧质量分",
    ]
    return metrics, highlights, []


def _summarize_generic(raw: Any, score: float | None) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    metrics = [_metric("得分", score, "score")]
    if is_dataclass(raw):
        fields = [name for name in raw.__dataclass_fields__.keys() if name not in {"applicable", "skip_reason"}]
        highlights = [f"字段: {', '.join(fields[:5])}"]
    else:
        highlights = ["无定制摘要，已返回基础得分"]
    return metrics, highlights, []


def _build_card(name: str, result: DimensionResult) -> dict[str, Any]:
    label = DIMENSION_CATALOG.get(name, {}).get("label", name)
    description = DIMENSION_CATALOG.get(name, {}).get("description", "")
    score = _float(result.score)
    if not result.applicable:
        return {
            "key": name,
            "label": label,
            "description": description,
            "applicable": False,
            "skip_reason": result.skip_reason or "not applicable",
            "score": score,
            "weight": _float(result.weight),
            "band": _score_band(score),
            "metrics": [],
            "highlights": [],
            "events": [],
        }

    raw = result.details
    summarizers = {
        "face_identity": _summarize_face_identity,
        "expression": _summarize_expression,
        "biological_anomaly": _summarize_biological,
        "motion_logic": _summarize_motion,
        "physics": _summarize_physics,
        "background": _summarize_background,
        "temporal_coherence": _summarize_temporal,
        "perceptual_quality": _summarize_perceptual,
    }
    metrics, highlights, events = summarizers.get(name, lambda item: _summarize_generic(item, score))(raw)
    return {
        "key": name,
        "label": label,
        "description": description,
        "applicable": True,
        "skip_reason": result.skip_reason,
        "score": score,
        "weight": _float(result.weight),
        "band": _score_band(score),
        "metrics": metrics,
        "highlights": highlights,
        "events": events,
    }


def build_dashboard_report(
    report: EvaluationReport,
    run_config: WebUIRunConfig,
    elapsed_sec: float,
) -> dict[str, Any]:
    selected_order = list(run_config.selected_dimensions)
    cards = [_build_card(name, report.dimensions[name]) for name in selected_order if name in report.dimensions]
    scored_cards = [card for card in cards if card["score"] is not None]
    best = max(scored_cards, key=lambda item: item["score"]) if scored_cards else None
    worst = min(scored_cards, key=lambda item: item["score"]) if scored_cards else None
    return {
        "video_name": Path(run_config.video_path).name,
        "video_path": run_config.video_path,
        "scope": run_config.scope,
        "device": run_config.device,
        "elapsed_sec": round(elapsed_sec, 3),
        "final_score": _float(report.final_score),
        "active_dimensions": report.active_dimensions,
        "selected_dimensions": selected_order,
        "video_processing": {
            "sample_stride": run_config.video_config.sample_stride,
            "max_frames": run_config.video_config.max_frames,
            "max_side": run_config.video_config.max_side,
            "parallel": run_config.parallel,
            "max_workers": run_config.max_workers,
            "enable_mllm": run_config.enable_mllm,
            "au_backend": run_config.au_backend,
            "au_external_python": run_config.au_external_python,
        },
        "summary": {
            "dimension_count": len(cards),
            "applicable_count": sum(1 for card in cards if card["applicable"]),
            "skipped_count": sum(1 for card in cards if not card["applicable"]),
            "best_dimension": best["label"] if best else None,
            "best_score": best["score"] if best else None,
            "worst_dimension": worst["label"] if worst else None,
            "worst_score": worst["score"] if worst else None,
        },
        "dimensions": cards,
    }
