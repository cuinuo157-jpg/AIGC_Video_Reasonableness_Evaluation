from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import cv2

from src.evaluation_pipeline import EvaluationReport
from src.expression_naturalness.au_rules import (
    AU_ACTIVATION_THRESHOLD,
    NATURAL_EXPRESSIONS,
    check_au_combination,
)
from src.mllm.prompts import MOTION_NATURALNESS_PROMPT
from src.mllm.prompts.physics_commonsense import build_physics_prompt

if TYPE_CHECKING:
    from .service import WebUIRunConfig


ArtifactMap = dict[str, list[dict[str, Any]]]


def generate_visual_artifacts(
    report: EvaluationReport,
    run_config: WebUIRunConfig,
    hub: Any,
    output_dir: Path,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[str, list[dict[str, Any]], ArtifactMap]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_artifacts: list[dict[str, Any]] = []
    by_dimension: ArtifactMap = {}
    video_name = Path(run_config.video_path).stem

    def append_log(message: str) -> None:
        if log_fn is not None:
            log_fn(message)

    def add_artifact(
        dimension: str,
        label: str,
        path: Path,
        kind: str = "file",
    ) -> None:
        record = {
            "dimension": dimension,
            "label": label,
            "path": str(path.resolve()),
            "kind": kind,
        }
        all_artifacts.append(record)
        by_dimension.setdefault(dimension, []).append(record)

    def safe_hub_get(name: str) -> Any:
        try:
            return hub.get(name)
        except Exception:
            return None

    def write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def collect_matching(dimension: str, dim_dir: Path, patterns: list[tuple[str, str]]) -> None:
        if not dim_dir.exists():
            return
        for label, pattern in patterns:
            for matched in sorted(dim_dir.glob(pattern)):
                kind = "directory" if matched.is_dir() else _guess_kind(matched)
                add_artifact(dimension, label, matched, kind)

    for dimension in run_config.selected_dimensions:
        result = report.dimensions.get(dimension)
        if result is None or not result.applicable:
            continue

        dim_dir = output_dir / dimension
        dim_dir.mkdir(parents=True, exist_ok=True)

        try:
            if dimension == "face_identity":
                frames = safe_hub_get("video_frames")
                frame_data = safe_hub_get("face_embedding")
                if frames and frame_data:
                    from scripts.debug_face_identity import save_visualization

                    save_visualization(frames, frame_data, result.details, str(dim_dir))
                    add_artifact(dimension, "可视化目录", dim_dir, "directory")
                    collect_matching(dimension, dim_dir, [("关键帧", "*.jpg")])

            elif dimension == "expression":
                frames = safe_hub_get("video_frames")
                au_per_frame = safe_hub_get("au_features")
                if not au_per_frame:
                    au_per_frame = _rebuild_au_per_frame(getattr(result.details, "au_sequences", {}) or {})
                if frames and au_per_frame:
                    from scripts.debug_expression import save_visualization

                    analysis = _build_expression_analysis(au_per_frame)
                    save_visualization(frames, au_per_frame, analysis, str(dim_dir))
                    add_artifact(dimension, "可视化目录", dim_dir, "directory")
                    collect_matching(dimension, dim_dir, [("关键帧", "*.jpg")])

            elif dimension == "biological_anomaly":
                frames = safe_hub_get("video_frames")
                keypoints_seq = safe_hub_get("keypoints")
                if frames and keypoints_seq:
                    from scripts.debug_bio_anomaly import save_visualization
                    from src.biological_anomaly.prompts import BIOLOGICAL_ANOMALY_PROMPT

                    l1 = {
                        "all": (
                            list(getattr(result.details, "eye_anomalies", []) or [])
                            + list(getattr(result.details, "mouth_anomalies", []) or [])
                            + list(getattr(result.details, "hand_anomalies", []) or [])
                            + list(getattr(result.details, "body_anomalies", []) or [])
                        )
                    }
                    l2 = {"all": []}
                    save_visualization(frames, keypoints_seq, l1, l2, str(dim_dir))
                    add_artifact(dimension, "可视化目录", dim_dir, "directory")
                    collect_matching(dimension, dim_dir, [("关键帧", "*.jpg")])
                    mllm_raw = getattr(result.details, "mllm_raw_result", None)
                    if mllm_raw is not None:
                        mllm_path = dim_dir / f"{video_name}_mllm_prompt_response.json"
                        write_json(mllm_path, {"prompt": BIOLOGICAL_ANOMALY_PROMPT, "response": mllm_raw})
                        add_artifact(dimension, "MLLM 原始输出", mllm_path, "json")

            elif dimension == "motion_logic":
                flows = safe_hub_get("raft_flow") or safe_hub_get("optical_flow")
                if flows:
                    from scripts.debug_dynamics import (
                        _collect_trajectory_events,
                        _save_trajectory_event_artifacts,
                        save_visualizations,
                    )

                    trajectories = safe_hub_get("tracking")
                    subject_result = safe_hub_get("subject_masks")
                    masks = getattr(subject_result, "masks", None) if subject_result is not None else None
                    save_visualizations(
                        flows,
                        result.details.dynamics_detail,
                        video_name,
                        dim_dir,
                        masks=masks,
                        trajectories=trajectories,
                        trajectory_detail=getattr(result.details, "trajectory_curvature_detail", None),
                    )
                    frames_bgr = safe_hub_get("video_frames") or []
                    if trajectories and frames_bgr:
                        frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
                        events = _collect_trajectory_events(trajectories)
                        _save_trajectory_event_artifacts(
                            frames_rgb,
                            trajectories,
                            events,
                            dim_dir,
                            video_name,
                            fps=24.0,
                            save_video=True,
                            save_events=True,
                        )
                    add_artifact(dimension, "可视化目录", dim_dir, "directory")
                    collect_matching(
                        dimension,
                        dim_dir,
                        [
                            ("光流图", "*_flow_*.png"),
                            ("主体光流图", "*_subject_*.png"),
                            ("主体流场图", "*_subject_flow_*.png"),
                            ("平均幅度热力图", "*_mean_mag.png"),
                            ("轨迹曲率图", "*_trajectory_curvature.png"),
                            ("轨迹曲率统计", "*_trajectory_curvature.json"),
                            ("轨迹叠加视频", "*_trajectory_overlay.mp4"),
                            ("轨迹事件", "*_trajectory_events.json"),
                            ("轨迹事件帧", "*_trajectory_event_*.jpg"),
                        ],
                    )
                    mllm_raw = getattr(result.details, "naturalness_mllm_result", None)
                    if mllm_raw is not None:
                        mllm_path = dim_dir / f"{video_name}_mllm_prompt_response.json"
                        write_json(mllm_path, {"prompt": MOTION_NATURALNESS_PROMPT, "response": mllm_raw})
                        add_artifact(dimension, "MLLM 原始输出", mllm_path, "json")

            elif dimension == "physics":
                mllm_raw = getattr(result.details, "vlm_raw_result", None)
                if mllm_raw is not None:
                    mllm_path = dim_dir / f"{video_name}_mllm_prompt_response.json"
                    write_json(
                        mllm_path,
                        {
                            "prompt": build_physics_prompt(
                                drift_events=getattr(result.details, "drift_events", None)
                            ),
                            "response": mllm_raw,
                        },
                    )
                    add_artifact(dimension, "VLM 原始输出", mllm_path, "json")

            elif dimension == "temporal_coherence":
                from scripts.debug_temporal_coherence import _save_detection_visualizations
                from src.temporal_coherence.analyzer import TemporalCoherenceAnalyzer

                vis_dir, n_saved = _save_detection_visualizations(
                    hub=hub,
                    analyzer=TemporalCoherenceAnalyzer(),
                    video_name=video_name,
                    device=run_config.device,
                    output_dir=dim_dir,
                )
                add_artifact(dimension, "检测可视化目录", vis_dir, "directory")
                collect_matching(
                    dimension,
                    vis_dir,
                    [("检测图", "*.jpg"), ("检测摘要", "*.json")],
                )
                append_log(f"[job] temporal_coherence 可视化已写出 {n_saved} 张检测图")
        except Exception as exc:
            append_log(f"[job] {dimension} 可视化生成失败: {exc}")

    return str(output_dir.resolve()), all_artifacts, by_dimension


def _guess_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image"
    if suffix in {".mp4", ".avi", ".mov"}:
        return "video"
    if suffix == ".json":
        return "json"
    return "file"


def _rebuild_au_per_frame(au_sequences: dict[str, list[float]]) -> list[dict[str, float]]:
    if not au_sequences:
        return []
    frame_count = max((len(seq) for seq in au_sequences.values()), default=0)
    frames: list[dict[str, float]] = []
    for idx in range(frame_count):
        frame_aus: dict[str, float] = {}
        for au_name, seq in au_sequences.items():
            if idx < len(seq):
                frame_aus[au_name] = float(seq[idx])
        frames.append(frame_aus)
    return frames


def _build_expression_analysis(au_per_frame: list[dict[str, float]]) -> dict[str, Any]:
    violation_frames: list[int] = []
    frame_expressions: list[list[str]] = []
    for idx, aus in enumerate(au_per_frame):
        if check_au_combination(aus):
            violation_frames.append(idx)
        active = {name for name, score in aus.items() if score >= AU_ACTIVATION_THRESHOLD}
        labels = []
        for expr_name, pattern in NATURAL_EXPRESSIONS.items():
            required_met = all(au in active for au in pattern["required"])
            forbidden_clear = all(au not in active for au in pattern["forbidden"])
            if required_met and forbidden_clear:
                labels.append(expr_name)
        frame_expressions.append(labels or ["neutral"])
    return {
        "violation_frames": violation_frames,
        "frame_expressions": frame_expressions,
    }
