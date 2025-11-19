#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# 添加项目根目录到路径
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)

from tqdm import tqdm  # type: ignore

VIDEO_EXTENSIONS = {".mp4"}

try:
    from src.temporal_reasoning.core.config import TemporalReasoningConfig, load_config_from_yaml
    from src.temporal_reasoning.utils.video_utils import get_video_info, load_video_frames
    from src.temporal_reasoning.region_analysis.pipeline import (
        RegionAnalysisPipeline,
        RegionAnalysisPipelineConfig,
        RegionDefinition,
    )
    from src.temporal_reasoning.region_analysis.region_temporal_change_detector import RegionTemporalChangeConfig
except ModuleNotFoundError as exc:  # pragma: no cover - defensive runtime guard
    raise RuntimeError(
        "无法导入 'src' 模块，请确认在项目根目录运行该脚本，"
        "或检查仓库结构是否保持默认布局。"
    ) from exc


def collect_video_paths(inputs: Sequence[str]) -> List[Path]:
    collected: List[Path] = []
    missing: List[str] = []
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            missing.append(raw)
            continue
        if path.is_dir():
            candidates = sorted(
                p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
            )
            collected.extend(candidates)
        else:
            collected.append(path)
    if missing:
        raise FileNotFoundError(
            "以下路径不存在，请检查: " + ", ".join(missing)
        )
    unique_ordered = list(dict.fromkeys(collected))
    if not unique_ordered:
        raise ValueError("未找到可处理的视频文件，请检查输入路径或目录内容是否包含受支持的格式。")
    return unique_ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run region flow/appearance change analysis on one or multiple videos."
    )
    parser.add_argument(
        "--video",
        required=True,
        nargs="+",
        help="Video file(s) or directory(ies). Directories will be scanned for .mp4 files only.",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["mouth"],
        help="Regions to analyze (default: mouth). Available: mouth, left_eye, right_eye, left_hand, right_hand",
    )
    parser.add_argument(
        "--config",
        help="Temporal reasoning YAML config. If omitted, default config is used.",
    )
    parser.add_argument(
        "--output",
        default="region_analysis_report.json",
        help="Path to save the analysis report (JSON). For batch mode this becomes a summary file.",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to store per-video reports when processing multiple videos (optional).",
    )
    parser.add_argument(
        "--min_area",
        type=int,
        help="Override minimum area (pixels) required for region masks.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        help="Override additional margin ratio around region polygons.",
    )
    parser.add_argument(
        "--motion_threshold",
        type=float,
        help="Override motion change threshold for regions.",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        help="Override color histogram similarity drop threshold.",
    )
    parser.add_argument(
        "--hist_diff_threshold",
        type=float,
        help="Override histogram similarity frame-to-frame difference threshold.",
    )
    parser.add_argument(
        "--consecutive_frames",
        type=int,
        help="Override required consecutive frames for anomaly.",
    )
    parser.add_argument(
        "--baseline_window",
        type=int,
        help="Override baseline window used to compute reference motion.",
    )
    parser.add_argument(
        "--disable_flow",
        action="store_true",
        help="Disable optical-flow change detection (only use appearance similarity).",
    )
    parser.add_argument(
        "--disable_color",
        action="store_true",
        help="Disable color histogram similarity check (only use motion).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save region mask visualizations for each frame.",
    )
    parser.add_argument(
        "--per_region_vis",
        action="store_true",
        help="Also save per-region visualization images (in addition to combined overlays).",
    )
    parser.add_argument(
        "--vis_dir",
        help="Directory to store visualization frames.",
    )
    parser.add_argument(
        "--vis_max_frames",
        type=int,
        default=150,
        help="Maximum number of visualization frames to save (0 for unlimited).",
    )
    parser.add_argument(
        "--debug_stats",
        help="Path to save per-frame statistics (JSON). For batch mode this becomes a directory unless --debug_dir is set.",
    )
    parser.add_argument(
        "--debug_dir",
        help="Directory to store per-video debug statistics when processing multiple videos.",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="详细输出模式：输出完整分析数据，包含所有详细信息和调试数据（默认：简化模式）",
    )
    return parser.parse_args()


def build_pipeline_config(
    temporal_config: TemporalReasoningConfig,
    regions: Sequence[str],
    min_area: Optional[int],
    margin: Optional[float],
    motion_threshold: Optional[float],
    similarity_threshold: Optional[float],
    hist_diff_threshold: Optional[float],
    consecutive_frames: Optional[int],
    baseline_window: Optional[int],
    disable_flow: bool,
    disable_color: bool,
    enable_visualization: bool,
    visualization_output_dir: Optional[str],
    visualization_max_frames: int,
    per_region_visualization: bool,
) -> RegionAnalysisPipelineConfig:
    # RegionAnalysisPipeline 需要 keypoint 配置（用于人体区域分析）
    # 如果 temporal_config.keypoint 为 None，则使用默认配置
    from src.temporal_reasoning.core.config import KeypointConfig
    keypoint_config = temporal_config.keypoint if temporal_config.keypoint is not None else KeypointConfig()
    
    pipeline_config = RegionAnalysisPipelineConfig(
        raft=temporal_config.raft,
        keypoint=keypoint_config,
        enable_visualization=enable_visualization,
        visualization_output_dir=visualization_output_dir,
        visualization_max_frames=visualization_max_frames,
        per_region_visualization=per_region_visualization,
    )

    available_regions = {region.name: copy.deepcopy(region) for region in pipeline_config.regions}
    selected_names = list(available_regions.keys()) if not regions else list(dict.fromkeys(regions))

    selected_definitions: List[RegionDefinition] = []
    for name in selected_names:
        if name not in available_regions:
            raise ValueError(f"Unknown region '{name}'. Available options: {', '.join(available_regions.keys())}")

        region_def = available_regions[name]
        config = copy.deepcopy(region_def.temporal_config)

        if motion_threshold is not None:
            config.motion_threshold = motion_threshold
        if similarity_threshold is not None:
            config.similarity_threshold = similarity_threshold
        if hist_diff_threshold is not None:
            config.hist_diff_threshold = hist_diff_threshold
        if consecutive_frames is not None:
            config.consecutive_frames = consecutive_frames
        if baseline_window is not None:
            config.baseline_window = baseline_window
        if disable_flow:
            config.use_flow_change = False
        if disable_color:
            config.use_color_similarity = False

        region_def.temporal_config = config

        if min_area is not None:
            region_def.min_area = min_area
        if margin is not None:
            region_def.margin_ratio = margin

        selected_definitions.append(region_def)

    pipeline_config.regions = selected_definitions
    return pipeline_config


def load_temporal_config(config_path: str | None) -> TemporalReasoningConfig:
    if config_path:
        return load_config_from_yaml(config_path)
    return TemporalReasoningConfig()


def run_analysis_for_video(
    pipeline: RegionAnalysisPipeline,
    video_path: Path,
    simple_output: bool = False,
) -> Dict[str, Any]:
    video_info = get_video_info(str(video_path))
    fps = float(video_info.get("fps") or 30.0)
    frames = load_video_frames(str(video_path))
    analysis_result = pipeline.analyze(frames, fps=fps, video_path=str(video_path), simple_output=simple_output)

    if simple_output:
        # 简化输出模式
        report: Dict[str, Any] = {
            "video_path": str(video_path),
            "has_anomaly": analysis_result.get("has_anomaly", False),
            "score": analysis_result.get("score", 1.0),
            "total_anomaly_count": analysis_result.get("total_anomaly_count", 0),
            "region_summary": analysis_result.get("region_summary", {}),
        }
    else:
        # 详细输出模式
        report: Dict[str, Any] = {
            "video_path": str(video_path),
            "fps": fps,
            "frame_count": len(frames),
            "anomaly_count": len(analysis_result.get("anomalies", [])),
            "analysis": analysis_result,
            "video_info": video_info,
        }
    return report


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (Path,)):
        return str(obj)
    return obj


def write_report(report: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(report), f, ensure_ascii=False, indent=2)


def write_debug_stats(report: Dict[str, Any], path: Optional[Path]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    region_stats = {
        name: region_data.get("metadata", {}).get("frame_stats", [])
        for name, region_data in report.get("analysis", {}).get("regions", {}).items()
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(region_stats), f, ensure_ascii=False, indent=2)
    print(f"[RegionAnalysis] 帧级统计已保存到: {path}")


def main() -> None:
    args = parse_args()
    video_paths = collect_video_paths(args.video)
    multi_video = len(video_paths) > 1

    temporal_config = load_temporal_config(args.config)
    pipeline_config = build_pipeline_config(
        temporal_config=temporal_config,
        regions=args.regions,
        min_area=args.min_area,
        margin=args.margin,
        motion_threshold=args.motion_threshold,
        similarity_threshold=args.similarity_threshold,
        hist_diff_threshold=args.hist_diff_threshold,
        consecutive_frames=args.consecutive_frames,
        baseline_window=args.baseline_window,
        disable_flow=args.disable_flow,
        disable_color=args.disable_color,
        enable_visualization=args.visualize,
        visualization_output_dir=args.vis_dir,
        visualization_max_frames=args.vis_max_frames,
        per_region_visualization=args.per_region_vis,
    )

    # 设置输出目录（多视频模式）
    if multi_video:
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else project_root / "outputs" / "region_analysis_reports"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置调试统计目录
        debug_dir = None
        if args.debug_dir:
            debug_dir = Path(args.debug_dir).expanduser().resolve()
        elif args.debug_stats:
            debug_dir = Path(args.debug_stats).expanduser().resolve()
        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
        debug_dir = None
    
    summary_records: List[Dict[str, Any]] = []

    pipeline = RegionAnalysisPipeline(pipeline_config)
    progress = tqdm(video_paths, desc="Processing videos", unit="video") if tqdm else video_paths

    for video_path in progress:
        report = run_analysis_for_video(pipeline, video_path, simple_output=args.simple)

        if multi_video:
            output_path = (output_dir / f"{video_path.stem}_region_analysis.json").resolve()
            debug_path = (debug_dir / f"{video_path.stem}_frame_stats.json").resolve() if debug_dir else None
        else:
            output_path = Path(args.output).expanduser().resolve()
            debug_path = Path(args.debug_stats).expanduser().resolve() if args.debug_stats else None

        write_report(report, output_path)
        if not args.simple:  # 只在详细模式下保存调试统计
            write_debug_stats(report, debug_path)

        anomaly_count = report.get("anomaly_count") or report.get("total_anomaly_count", 0)
        if multi_video:
            summary_record = {
                "video_path": str(video_path),
                "report_path": str(output_path),
                "anomaly_count": anomaly_count,
                "has_anomaly": report.get("has_anomaly", False) if args.simple else (anomaly_count > 0),
                "score": report.get("score") or report.get("analysis", {}).get("score", 1.0),
            }
            if not args.simple:
                summary_record["fps"] = report.get("fps")
                summary_record["frame_count"] = report.get("frame_count")
            if debug_path is not None:
                summary_record["debug_stats_path"] = str(debug_path)
            summary_records.append(summary_record)

        if tqdm:
            progress.set_postfix({"video": video_path.name, "anomalies": anomaly_count})
        print(f"[RegionAnalysis] 分析完成，报告已保存到: {output_path}")

    if multi_video:
        summary_output = Path(args.output).expanduser().resolve()
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        total_anomalies = sum(int(record.get("anomaly_count") or 0) for record in summary_records)
        batch_summary = {
            "video_count": len(summary_records),
            "total_anomaly_count": total_anomalies,
            "videos": summary_records,
        }
        with summary_output.open("w", encoding="utf-8") as f:
            json.dump(to_serializable(batch_summary), f, ensure_ascii=False, indent=2)
        print(
            f"[RegionAnalysis] 批量分析完成，共处理 {len(summary_records)} 个视频。"
            f" 汇总报告已保存到: {summary_output}"
        )


if __name__ == "__main__":
    main()
