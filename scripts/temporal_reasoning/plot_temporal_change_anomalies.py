#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化步骤1的区域时序变化异常（all_anomalies）
按对象分组显示
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 添加项目根目录到路径
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_temporal_change_anomalies(result_path: Path) -> Dict:
    """
    从分析结果中加载步骤1的区域时序变化异常
    
    Returns:
        包含异常数据和阈值的数据字典
    """
    data = json.loads(result_path.read_text(encoding="utf-8"))
    
    fps = data.get("fps") or data.get("video_info", {}).get("fps", 30.0)
    
    # 提取时序变化异常（步骤1的结构分析中的区域时序变化异常）
    temporal_change_anomalies = []
    if "anomalies" in data:
        all_anomalies = data["anomalies"]
        print(f"[调试] 总异常数量: {len(all_anomalies)}")
        
        for anomaly in all_anomalies:
            if isinstance(anomaly, dict):
                anomaly_type = anomaly.get("type", "")
                anomaly_type_lower = anomaly_type.lower()
                
                # 匹配逻辑：检查是否包含关键词
                # 注意：structural_region_temporal_change 是完整的类型名称
                is_temporal_change = (
                    "structural_region_temporal_change" in anomaly_type_lower or 
                    "region_temporal" in anomaly_type_lower or
                    anomaly_type == "structural_region_temporal_change"  # 精确匹配
                )
                
                if is_temporal_change:
                    temporal_change_anomalies.append(anomaly)
                    print(f"[匹配] 找到时序变化异常: type={anomaly_type}")
    
    # 调试信息：打印找到的异常类型
    if temporal_change_anomalies:
        print(f"[调试] 找到 {len(temporal_change_anomalies)} 个时序变化异常")
        # 打印前几个异常的类型和metadata信息
        for i, anomaly in enumerate(temporal_change_anomalies[:3]):
            metadata = anomaly.get('metadata', {})
            print(f"  异常 {i+1}: type={anomaly.get('type')}, "
                  f"confidence={anomaly.get('confidence', 'N/A')}, "
                  f"has_metadata={bool(metadata)}, "
                  f"object_id={metadata.get('object_id') if isinstance(metadata, dict) else None}")
    else:
        # 如果没有找到，打印所有异常类型用于调试
        if "anomalies" in data:
            all_types = [a.get("type", "unknown") for a in data["anomalies"] if isinstance(a, dict)]
            unique_types = list(set(all_types))
            print(f"[调试] 未找到时序变化异常。")
            print(f"      所有异常类型 ({len(unique_types)} 种): {unique_types}")
            print(f"      提示：检查异常类型是否包含 'structural_region_temporal_change' 或 'region_temporal'")
            
            # 检查是否有类似的类型
            similar_types = [t for t in unique_types if "temporal" in t.lower() or "region" in t.lower() or "structural" in t.lower()]
            if similar_types:
                print(f"      发现相似类型: {similar_types}")
    
    # 按对象分组异常
    anomalies_by_object: Dict[int, Dict[str, Any]] = {}
    anomalies_without_object_id = []  # 记录没有object_id的异常
    
    for anomaly in temporal_change_anomalies:
        if isinstance(anomaly, dict):
            metadata = anomaly.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            
            object_id = metadata.get("object_id")
            if object_id is not None:
                if object_id not in anomalies_by_object:
                    anomalies_by_object[object_id] = {
                        "anomalies": [],
                        "class_name": metadata.get("class_name", ""),
                        "baseline_motion": None,
                    }
                anomalies_by_object[object_id]["anomalies"].append(anomaly)
                # 提取baseline（从第一个异常的metadata中）
                if anomalies_by_object[object_id]["baseline_motion"] is None:
                    anomalies_by_object[object_id]["baseline_motion"] = metadata.get("baseline_motion")
            else:
                # 如果没有object_id，记录但不分组（可能是融合后的异常丢失了metadata）
                anomalies_without_object_id.append(anomaly)
    
    # 如果有异常但没有object_id，给出提示
    if anomalies_without_object_id:
        print(f"[警告] 有 {len(anomalies_without_object_id)} 个时序变化异常缺少 object_id，无法按对象分组")
        print(f"      这可能是因为异常融合时丢失了 metadata 字段")
    
    # 提取阈值信息
    temporal_threshold = None
    similarity_threshold = None
    hist_diff_threshold = None
    
    if "thresholds" in data and isinstance(data["thresholds"], dict):
        region_temporal = data["thresholds"].get("region_temporal", {})
        if region_temporal:
            temporal_threshold = region_temporal.get("motion_threshold")
            similarity_threshold = region_temporal.get("similarity_threshold")
            hist_diff_threshold = region_temporal.get("hist_diff_threshold")
    
    return {
        "fps": fps,
        "video_path": data.get("video_path") or data.get("video_info", {}).get("path", ""),
        "temporal_change_anomalies": temporal_change_anomalies,
        "anomalies_by_object": anomalies_by_object,
        "motion_threshold": temporal_threshold,
        "similarity_threshold": similarity_threshold,
        "hist_diff_threshold": hist_diff_threshold,
    }


def plot_temporal_change_anomalies(
    anomalies_by_object: Dict[int, Dict[str, Any]],
    motion_threshold: Optional[float],
    similarity_threshold: Optional[float],
    hist_diff_threshold: Optional[float],
    fps: float,
    save_path: Optional[Path] = None,
    show: bool = True,
    title: Optional[str] = None,
) -> None:
    """
    绘制步骤1的区域时序变化异常可视化（按对象分组）
    
    Args:
        anomalies_by_object: 按对象分组的异常字典
        motion_threshold: 运动阈值
        similarity_threshold: 相似度阈值
        hist_diff_threshold: 直方图差异阈值
        fps: 帧率
        save_path: 保存路径
        show: 是否显示
        title: 图表标题
    """
    if not anomalies_by_object:
        print("[警告] 没有找到区域时序变化异常数据")
        return
    
    # 计算需要的子图数量（每个对象一个子图，最多显示前10个对象）
    object_ids = sorted(anomalies_by_object.keys())
    max_objects = min(10, len(object_ids))
    object_ids = object_ids[:max_objects]
    
    if len(object_ids) == 0:
        print("[警告] 没有可显示的对象")
        return
    
    # 创建子图（每行2个）
    n_cols = 2
    n_rows = (max_objects + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharex=True)
    
    # 处理axes的维度
    if n_rows == 1:
        if isinstance(axes, np.ndarray):
            axes = axes.reshape(1, -1)
        else:
            axes = np.array([[axes[0], axes[1]]])
    elif not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    
    axes_flat = axes.flatten()
    
    for idx, obj_id in enumerate(object_ids):
        if idx >= len(axes_flat):
            break
        
        ax = axes_flat[idx]
        obj_data = anomalies_by_object[obj_id]
        obj_anomalies = obj_data["anomalies"]
        class_name = obj_data["class_name"]
        baseline_motion = obj_data["baseline_motion"]
        
        # 提取该对象异常的时间戳和数值
        timestamps = []
        motion_values = []
        motion_changes = []
        
        for anomaly in obj_anomalies:
            if isinstance(anomaly, dict):
                timestamp = anomaly.get("timestamp")
                if timestamp is not None:
                    timestamps.append(float(timestamp))
                    metadata = anomaly.get("metadata", {})
                    motion_values.append(metadata.get("motion_value", 0))
                    motion_changes.append(metadata.get("motion_change", 0))
        
        if not timestamps:
            ax.text(0.5, 0.5, f"Object {obj_id}\nNo anomalies", 
                   transform=ax.transAxes, ha="center", va="center", fontsize=10)
            obj_label = f"Object {obj_id}"
            if class_name:
                obj_label += f" ({class_name})"
            ax.set_title(obj_label, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # 绘制motion_value和motion_change散点
        if motion_values:
            ax.scatter(timestamps, motion_values, 
                      color="tab:blue", s=60, alpha=0.7, label="Motion Value", zorder=2)
        
        if motion_changes:
            ax.scatter(timestamps, motion_changes, 
                      color="tab:orange", s=60, alpha=0.7, label="Motion Change", zorder=2)
        
        # 标注baseline
        if baseline_motion is not None:
            ax.axhline(y=baseline_motion, color="green", linestyle="--", 
                      label=f"Baseline: {baseline_motion:.2f}", linewidth=1.5, alpha=0.8)
        
        # 标注阈值（baseline ± threshold）
        if baseline_motion is not None and motion_threshold is not None:
            upper_threshold = baseline_motion + motion_threshold
            lower_threshold = baseline_motion - motion_threshold
            ax.axhline(y=upper_threshold, color="red", linestyle="--", 
                      label=f"Threshold: ±{motion_threshold:.2f}", linewidth=1, alpha=0.7)
            ax.axhline(y=lower_threshold, color="red", linestyle="--", linewidth=1, alpha=0.7)
        
        # 标记异常点（红色X，覆盖在散点上）
        if timestamps:
            if motion_values:
                ax.scatter(timestamps, motion_values, 
                          color="red", s=150, marker="X", 
                          label=f"Anomalies ({len(obj_anomalies)})", 
                          zorder=5, edgecolors="darkred", linewidths=1.5)
            elif motion_changes:
                ax.scatter(timestamps, motion_changes, 
                          color="red", s=150, marker="X", 
                          label=f"Anomalies ({len(obj_anomalies)})", 
                          zorder=5, edgecolors="darkred", linewidths=1.5)
        
        # 设置标题和标签
        obj_label = f"Object {obj_id}"
        if class_name:
            obj_label += f" ({class_name})"
        ax.set_title(obj_label, fontsize=11, fontweight="bold")
        ax.set_ylabel("Motion Value/Change", fontsize=9)
        if idx < len(axes_flat) - n_cols:  # 不是最后一行
            ax.set_xticklabels([])
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(len(object_ids), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    # 设置共享的x轴标签（只在最后一行显示）
    for idx in range(max(0, len(object_ids) - n_cols), len(object_ids)):
        if idx < len(axes_flat):
            axes_flat[idx].set_xlabel("Time (s)", fontsize=10)
    
    # 添加统计信息
    total_anomalies = sum(len(obj_data["anomalies"]) for obj_data in anomalies_by_object.values())
    stats_text = (
        f"Total Objects: {len(anomalies_by_object)}\n"
        f"Total Anomalies: {total_anomalies}\n"
        f"Displayed Objects: {len(object_ids)}\n"
    )
    if motion_threshold is not None:
        stats_text += f"\nMotion Threshold: {motion_threshold:.2f}"
    if similarity_threshold is not None:
        stats_text += f"\nSimilarity Threshold: {similarity_threshold:.2f}"
    if hist_diff_threshold is not None:
        stats_text += f"\nHist Diff Threshold: {hist_diff_threshold:.3f}"
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    else:
        fig.suptitle("Temporal Change Anomalies by Object (Step 1: Region Temporal Analysis)", 
                    fontsize=14, fontweight="bold", y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        save_path = save_path.expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] 区域时序变化异常可视化已保存到: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="可视化步骤1的区域时序变化异常（按对象分组）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本使用（显示图表）
  python plot_temporal_change_anomalies.py --result outputs/temporal_reasoning/video_result.json
  
  # 保存图像到文件
  python plot_temporal_change_anomalies.py --result outputs/temporal_reasoning/video_result.json \\
      --output outputs/temporal_anomalies_plot.png --no-show
        """
    )
    parser.add_argument("--result", required=True, help="分析结果JSON文件路径（run_analysis.py的输出）")
    parser.add_argument("--output", help="可选：保存图像的路径（png/jpg等）")
    parser.add_argument("--no-show", action="store_true", help="仅保存图像，不在窗口展示")
    parser.add_argument("--title", help="图像标题，例如视频名")
    return parser.parse_args()


def main():
    args = parse_args()
    result_path = Path(args.result).expanduser().resolve()
    
    if not result_path.exists():
        raise FileNotFoundError(f"结果文件不存在: {result_path}")
    
    payload = load_temporal_change_anomalies(result_path)
    
    if not payload["temporal_change_anomalies"]:
        print("[警告] 没有找到区域时序变化异常，请确认分析结果中包含了步骤1的区域时序变化检测结果。")
        print("      提示：请确认运行 run_analysis.py 时启用了区域时序变化检测（默认启用）")
        return
    
    # 检查是否有按对象分组的异常
    if not payload["anomalies_by_object"]:
        print("[错误] 找到时序变化异常，但无法按对象分组（缺少 object_id）。")
        print("      这通常是因为异常融合时丢失了 metadata 字段。")
        print("      解决方案：")
        print("      1. 使用最新版本的代码（已修复融合时保留 metadata 的问题）")
        print("      2. 或者检查分析结果中的异常是否包含 metadata.object_id 字段")
        print(f"      找到的异常数量: {len(payload['temporal_change_anomalies'])}")
        return
    
    default_title = (
        f"{Path(payload['video_path']).name} - Temporal Change Anomalies"
        if payload["video_path"]
        else "Temporal Change Anomalies Analysis"
    )
    title = args.title or default_title
    
    plot_temporal_change_anomalies(
        anomalies_by_object=payload["anomalies_by_object"],
        motion_threshold=payload["motion_threshold"],
        similarity_threshold=payload["similarity_threshold"],
        hist_diff_threshold=payload["hist_diff_threshold"],
        fps=payload["fps"],
        save_path=Path(args.output).expanduser().resolve() if args.output else None,
        show=not args.no_show,
        title=title,
    )


if __name__ == "__main__":
    main()

