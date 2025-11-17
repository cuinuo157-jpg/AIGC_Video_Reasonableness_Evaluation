#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从区域时序分析报告中可视化光流与颜色指标。
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_report(report_path: Path, region: str = "motion_region") -> dict:
    """
    读取 JSON 报告并返回指定区域的数据
    
    支持两种报告格式：
    1. run_region_temporal_analysis.py 的输出：analysis.regions[region].metadata.frame_stats
    2. run_analysis.py 的输出：motion_metrics.frame_stats
    """
    data = json.loads(report_path.read_text(encoding="utf-8"))
    analysis = data.get("analysis", {})
    regions = analysis.get("regions", {})

    # 格式1: run_region_temporal_analysis.py 的输出格式
    if regions:
        if region not in regions:
            raise ValueError(
                f"报告中无区域 '{region}'，可用区域: {', '.join(sorted(regions.keys()))}"
            )
        metadata = regions[region].get("metadata", {})
        frame_stats = metadata.get("frame_stats", [])
    # 格式2: run_analysis.py 的输出格式（motion_metrics）
    elif "motion_metrics" in data:
        metadata = data.get("motion_metrics", {})
        frame_stats = metadata.get("frame_stats", [])
        if not frame_stats:
            raise ValueError(
                f"{report_path} 中没有 frame_stats 数据。"
                "请确认分析时使用了mask（结构分析）才能生成运动检测的frame_stats。"
            )
    # 格式3: 旧格式（兼容性）
    else:
        metadata = analysis.get("metadata", {})
        frame_stats = metadata.get("frame_stats", [])
        if region != "mouth":
            raise ValueError("报告为旧格式，仅包含 mouth 区域，请指定 --region mouth。")

    if not frame_stats:
        raise ValueError(
            f"{report_path} 中区域 '{region}' 没有 frame_stats 数据，请确认分析时是否开启 --debug_stats。"
        )
    
    # 从 video_info 或顶层获取 fps
    fps = data.get("fps") or data.get("video_info", {}).get("fps", 0.0)
    
    return {
        "frame_stats": frame_stats,
        "motion_threshold": metadata.get("motion_threshold", 0.0),
        "similarity_threshold": metadata.get("similarity_threshold", 0.0),
        "hist_diff_threshold": metadata.get("hist_diff_threshold"),
        "baseline_motion": metadata.get("baseline_motion", 0.0),
        "fps": fps,
        "video_path": data.get("video_path") or data.get("video_info", {}).get("path", ""),
        "region": region,
    }


def plot_metrics(
    frame_stats: list,
    motion_threshold: float,
    similarity_threshold: float,
    hist_diff_threshold: Optional[float],
    baseline_motion: float,
    save_path: Optional[Path] = None,
    show: bool = True,
    title: Optional[str] = None,
) -> None:
    """绘制光流 / 颜色指标曲线"""
    df = pd.DataFrame(frame_stats)
    if "timestamp" not in df.columns:
        raise ValueError("frame_stats 缺少 timestamp 字段；请使用新版检测脚本重新生成报告。")
    df["time"] = df["timestamp"]

    has_hist_diff = "hist_diff" in df.columns
    n_rows = 4 if has_hist_diff else 3
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(10, 12 if has_hist_diff else 10),
        sharex=True,
    )
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten().tolist()

    # 1. 光流幅值
    axes[0].plot(df["time"], df["motion_value"], label="motion_value")
    axes[0].axhline(baseline_motion, color="gray", linestyle="--", label="baseline_motion")
    axes[0].set_ylabel("Motion Value")
    axes[0].legend()
    axes[0].grid(True)

    # 2. 光流变化量
    axes[1].plot(df["time"], df["motion_change"], label="motion_change", color="tab:orange")
    axes[1].axhline(motion_threshold, color="red", linestyle="--", label="motion_threshold")
    axes[1].set_ylabel("Motion Change")
    axes[1].legend()
    axes[1].grid(True)

    # 3. 颜色相似度
    axes[2].plot(df["time"], df["hist_similarity"], label="hist_similarity", color="tab:green")
    axes[2].axhline(1 - similarity_threshold, color="purple", linestyle="--", label="1 - similarity_threshold")
    axes[2].set_ylabel("Hist Similarity")
    axes[2].legend()
    axes[2].grid(True)

    if has_hist_diff:
        axes[3].plot(df["time"], df["hist_diff"], label="hist_diff", color="tab:blue")
        if hist_diff_threshold is not None:
            axes[3].axhline(hist_diff_threshold, color="red", linestyle="--", label="hist_diff_threshold")
        axes[3].set_ylabel("Hist Diff")
        axes[3].legend()
        axes[3].grid(True)

    axes[-1].set_xlabel("Time (s)")

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if save_path:
        save_path = save_path.expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[plot] 曲线已保存到: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化区域分析报告中的光流与颜色指标")
    parser.add_argument("--report", required=True, help="region_analysis_report.json 路径")
    parser.add_argument("--output", help="可选：保存图像的路径（png/jpg 等）")
    parser.add_argument("--no-show", action="store_true", help="仅保存图像，不在窗口展示")
    parser.add_argument("--title", help="图像标题，例如视频名")
    parser.add_argument(
        "--region",
        default="motion_region",
        help="要可视化的区域名称（默认: motion_region，用于run_analysis.py的输出；"
             "对于run_region_temporal_analysis.py的输出，使用mouth、left_eye等）。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    report_path = Path(args.report).expanduser().resolve()
    payload = load_report(report_path, args.region)

    default_title = (
        f"{Path(payload['video_path']).name} [{payload['region']}]"
        if payload["video_path"]
        else payload["region"]
    )
    title = args.title or default_title
    plot_metrics(
        frame_stats=payload["frame_stats"],
        motion_threshold=payload["motion_threshold"],
        similarity_threshold=payload["similarity_threshold"],
        hist_diff_threshold=payload.get("hist_diff_threshold"),
        baseline_motion=payload["baseline_motion"],
        save_path=Path(args.output).expanduser().resolve() if args.output else None,
        show=not args.no_show,
        title=title,
    )


if __name__ == "__main__":
    main()