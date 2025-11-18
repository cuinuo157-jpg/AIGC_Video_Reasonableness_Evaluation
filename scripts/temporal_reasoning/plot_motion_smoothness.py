#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化基于光流的运动平滑度
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 添加项目根目录到路径（虽然这个脚本不直接导入src模块，但保持一致性）
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_anomaly_data(result_path: Path) -> Dict:
    """
    从分析结果中加载异常数据和frame_stats
    
    Returns:
        包含时序变化异常和运动异常的数据字典
    """
    data = json.loads(result_path.read_text(encoding="utf-8"))
    
    fps = data.get("fps") or data.get("video_info", {}).get("fps", 30.0)
    
    # 提取时序变化异常（步骤1的结构分析中的区域时序变化异常）
    temporal_change_anomalies = []
    if "anomalies" in data:
        for anomaly in data["anomalies"]:
            if isinstance(anomaly, dict):
                anomaly_type = anomaly.get("type", "").lower()
                if "structural_region_temporal_change" in anomaly_type or "region_temporal" in anomaly_type:
                    temporal_change_anomalies.append(anomaly)
    
    # 按对象分组时序变化异常（用于详细可视化）
    temporal_anomalies_by_object = {}
    for anomaly in temporal_change_anomalies:
        if isinstance(anomaly, dict):
            metadata = anomaly.get("metadata", {})
            object_id = metadata.get("object_id")
            if object_id is not None:
                if object_id not in temporal_anomalies_by_object:
                    temporal_anomalies_by_object[object_id] = {
                        "anomalies": [],
                        "class_name": metadata.get("class_name", ""),
                    }
                temporal_anomalies_by_object[object_id]["anomalies"].append(anomaly)
    
    # 提取运动异常（步骤2的光流分析中的异常）
    motion_anomalies = []
    if "anomalies" in data:
        for anomaly in data["anomalies"]:
            if isinstance(anomaly, dict):
                anomaly_type = anomaly.get("type", "").lower()
                if "region_flow_change" in anomaly_type or ("motion" in anomaly_type and "structural" not in anomaly_type):
                    motion_anomalies.append(anomaly)
    
    # 提取运动异常的frame_stats（从motion_metrics中）
    motion_frame_stats = []
    motion_baseline = None
    motion_threshold = None
    similarity_threshold = None
    hist_diff_threshold = None
    
    if "motion_metrics" in data:
        motion_metrics = data["motion_metrics"]
        motion_frame_stats = motion_metrics.get("frame_stats", [])
        motion_baseline = motion_metrics.get("baseline_motion")
        motion_threshold = motion_metrics.get("motion_threshold")
        similarity_threshold = motion_metrics.get("similarity_threshold")
        hist_diff_threshold = motion_metrics.get("hist_diff_threshold")
    
    # 提取时序变化异常的frame_stats（从analysis.regions中获取）
    # 注意：步骤1的区域时序变化异常可能没有统一的frame_stats，因为是对每个对象分别检测的
    temporal_frame_stats = []
    temporal_baseline = None
    temporal_threshold = None
    
    # 尝试从analysis.regions中提取frame_stats（合并所有区域的frame_stats）
    analysis = data.get("analysis", {})
    regions = analysis.get("regions", {})
    if regions:
        # 收集所有区域的frame_stats
        all_region_frame_stats = []
        for region_name, region_data in regions.items():
            if isinstance(region_data, dict):
                metadata = region_data.get("metadata", {})
                region_frame_stats = metadata.get("frame_stats", [])
                if region_frame_stats:
                    all_region_frame_stats.extend(region_frame_stats)
                    # 从第一个区域获取baseline和阈值（如果还没有）
                    if temporal_baseline is None:
                        temporal_baseline = metadata.get("baseline_motion")
                    if temporal_threshold is None:
                        temporal_threshold = metadata.get("motion_threshold")
        
        # 如果有多个区域的frame_stats，按timestamp合并（取平均值）
        if all_region_frame_stats:
            try:
                df = pd.DataFrame(all_region_frame_stats)
                if 'timestamp' in df.columns and len(df) > 0:
                    # 按timestamp分组，取平均值
                    agg_dict = {
                        'motion_value': 'mean',
                        'motion_change': 'mean',
                        'hist_similarity': 'mean',
                    }
                    if 'hist_diff' in df.columns:
                        agg_dict['hist_diff'] = 'mean'
                    df_grouped = df.groupby('timestamp').agg(agg_dict).reset_index()
                    temporal_frame_stats = df_grouped.to_dict('records')
                else:
                    temporal_frame_stats = all_region_frame_stats
            except Exception as e:
                print(f"[警告] 合并区域frame_stats失败: {e}，使用原始数据")
                temporal_frame_stats = all_region_frame_stats
    
    # 如果还是没有找到，尝试从structure_metrics中获取
    if not temporal_frame_stats and "structure_metrics" in data:
        structure_metrics = data["structure_metrics"]
        if isinstance(structure_metrics, dict):
            temporal_frame_stats = structure_metrics.get("frame_stats", [])
            if temporal_baseline is None:
                temporal_baseline = structure_metrics.get("baseline_motion")
            if temporal_threshold is None:
                temporal_threshold = structure_metrics.get("motion_threshold")
    
    # 尝试从thresholds中获取区域时序变化的阈值
    if "thresholds" in data and isinstance(data["thresholds"], dict):
        region_temporal = data["thresholds"].get("region_temporal", {})
        if region_temporal:
            if temporal_threshold is None:
                temporal_threshold = region_temporal.get("motion_threshold")
            if temporal_baseline is None:
                temporal_baseline = region_temporal.get("baseline_motion")
    
    return {
        "fps": fps,
        "video_path": data.get("video_path") or data.get("video_info", {}).get("path", ""),
        "temporal_change_anomalies": temporal_change_anomalies,
        "motion_anomalies": motion_anomalies,
        "motion_frame_stats": motion_frame_stats,
        "motion_baseline": motion_baseline,
        "motion_threshold": motion_threshold,
        "similarity_threshold": similarity_threshold,
        "hist_diff_threshold": hist_diff_threshold,
        "temporal_frame_stats": temporal_frame_stats,
        "temporal_baseline": temporal_baseline,
        "temporal_threshold": temporal_threshold,
        "temporal_anomalies_by_object": temporal_anomalies_by_object,
    }


def load_smoothness_data(result_path: Path) -> Dict:
    """
    从分析结果中加载平滑度数据和异常信息
    
    支持两种格式：
    1. run_analysis.py 的输出：从motion_metrics或metadata中提取
    2. 直接包含smoothness_scores的JSON文件
    """
    data = json.loads(result_path.read_text(encoding="utf-8"))
    
    # 尝试从不同位置提取平滑度数据
    smoothness_scores = None
    timestamps = None
    fps = None
    
    # 方式1: 从motion_metrics中提取（run_analysis.py的输出格式）
    if "motion_metrics" in data:
        motion_metrics = data["motion_metrics"]
        if "smoothness_scores" in motion_metrics:
            smoothness_scores = motion_metrics["smoothness_scores"]
            if "smoothness_timestamps" in motion_metrics:
                timestamps = motion_metrics["smoothness_timestamps"]
            fps = data.get("fps") or data.get("video_info", {}).get("fps", 30.0)
    
    # 方式2: 从顶层metadata中提取
    if smoothness_scores is None and "metadata" in data:
        metadata = data["metadata"]
        if "smoothness_scores" in metadata:
            smoothness_scores = metadata["smoothness_scores"]
            if "smoothness_timestamps" in metadata:
                timestamps = metadata["smoothness_timestamps"]
            fps = metadata.get("fps", 30.0)
    
    # 方式3: 直接从顶层提取
    if smoothness_scores is None and "smoothness_scores" in data:
        smoothness_scores = data["smoothness_scores"]
        if "smoothness_timestamps" in data:
            timestamps = data["smoothness_timestamps"]
        elif "timestamps" in data:
            timestamps = data["timestamps"]
        fps = data.get("fps", 30.0)
    
    # 如果没有找到平滑度数据，尝试从sub_scores中提取
    if smoothness_scores is None and "sub_scores" in data:
        sub_scores = data["sub_scores"]
        if "motion_smoothness" in sub_scores:
            smoothness_scores = sub_scores["motion_smoothness"]
            # 需要根据视频信息生成时间戳
            fps = data.get("fps") or data.get("video_info", {}).get("fps", 30.0)
            if timestamps is None and fps > 0:
                timestamps = [i / fps for i in range(len(smoothness_scores))]
    
    if smoothness_scores is None:
        raise ValueError(
            f"{result_path} 中没有找到平滑度数据。\n"
            "请确认分析结果中包含了 motion_metrics.smoothness_scores。\n"
            "提示：运行 run_analysis.py 时会自动计算并保存平滑度数据。"
        )
    
    # 如果没有时间戳，根据fps生成
    if timestamps is None:
        fps = fps or data.get("fps") or data.get("video_info", {}).get("fps", 30.0)
        timestamps = [i / fps for i in range(len(smoothness_scores))]
    
    # 提取异常信息
    anomalies = []
    if "anomalies" in data:
        # 过滤出运动相关的异常
        for anomaly in data["anomalies"]:
            if isinstance(anomaly, dict):
                anomaly_type = anomaly.get("type", "")
                if "motion" in anomaly_type.lower() or "flow" in anomaly_type.lower():
                    anomalies.append(anomaly)
    
    # 提取阈值信息
    thresholds = {}
    if "motion_metrics" in data:
        motion_metrics = data["motion_metrics"]
        if "motion_threshold" in motion_metrics:
            thresholds["motion_threshold"] = motion_metrics["motion_threshold"]
        if "similarity_threshold" in motion_metrics:
            thresholds["similarity_threshold"] = motion_metrics["similarity_threshold"]
        if "hist_diff_threshold" in motion_metrics:
            thresholds["hist_diff_threshold"] = motion_metrics["hist_diff_threshold"]
    
    # 从thresholds字段提取（如果存在）
    if "thresholds" in data:
        thresholds_info = data["thresholds"]
        if isinstance(thresholds_info, dict):
            if "motion_discontinuity_threshold" in thresholds_info:
                thresholds["motion_threshold"] = thresholds_info["motion_discontinuity_threshold"]
    
    return {
        "smoothness_scores": smoothness_scores,
        "timestamps": timestamps,
        "fps": fps,
        "video_path": data.get("video_path") or data.get("video_info", {}).get("path", ""),
        "anomalies": anomalies,
        "thresholds": thresholds,
    }


def plot_anomalies(
    temporal_change_anomalies: List[Dict],
    motion_anomalies: List[Dict],
    motion_frame_stats: List[Dict],
    motion_baseline: Optional[float],
    motion_threshold: Optional[float],
    similarity_threshold: Optional[float],
    hist_diff_threshold: Optional[float],
    temporal_threshold: Optional[float],
    temporal_frame_stats: Optional[List[Dict]] = None,
    temporal_baseline: Optional[float] = None,
    fps: float = 30.0,
    save_path: Optional[Path] = None,
    show: bool = True,
    title: Optional[str] = None,
) -> None:
    """
    绘制时序变化异常和运动异常的可视化
    
    Args:
        temporal_change_anomalies: 时序变化异常列表（步骤1）
        motion_anomalies: 运动异常列表（步骤2）
        motion_frame_stats: 运动分析的frame_stats（用于下图）
        motion_baseline: 运动分析的baseline_motion
        motion_threshold: 运动阈值
        similarity_threshold: 相似度阈值
        hist_diff_threshold: 直方图差异阈值
        temporal_threshold: 时序变化异常阈值
        temporal_frame_stats: 时序变化异常的frame_stats（用于上图，如果为None则使用motion_frame_stats）
        temporal_baseline: 时序变化异常的baseline（如果为None则使用motion_baseline）
        fps: 帧率
        save_path: 保存路径
        show: 是否显示
        title: 图表标题
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    # ========== 上图：时序变化异常 ==========
    ax1 = axes[0]
    
    # 提取时序变化异常的时间戳和motion_value
    temporal_timestamps = []
    temporal_motion_values = []
    temporal_motion_changes = []
    temporal_baseline = None
    
    if temporal_change_anomalies:
        for anomaly in temporal_change_anomalies:
            if isinstance(anomaly, dict):
                timestamp = anomaly.get("timestamp")
                if timestamp is not None:
                    temporal_timestamps.append(float(timestamp))
                    # 从metadata中提取motion_value和motion_change
                    metadata = anomaly.get("metadata", {})
                    temporal_motion_values.append(metadata.get("motion_value", 0))
                    temporal_motion_changes.append(metadata.get("motion_change", 0))
                    # 提取baseline（每个异常可能有自己的baseline）
                    if temporal_baseline is None:
                        temporal_baseline = metadata.get("baseline_motion")
    
    # 如果没有从异常中提取到baseline，使用传入的temporal_baseline或motion_baseline
    if temporal_baseline is None:
        temporal_baseline = motion_baseline
    
    # 使用时序变化异常的frame_stats绘制曲线（优先使用temporal_frame_stats）
    frame_stats_to_plot = temporal_frame_stats if temporal_frame_stats else motion_frame_stats
    if frame_stats_to_plot:
        df = pd.DataFrame(frame_stats_to_plot)
        if "timestamp" in df.columns:
            if "motion_value" in df.columns:
                ax1.plot(df["timestamp"], df["motion_value"], 
                        label="Motion Value", color="tab:blue", linewidth=2, alpha=0.7)
            if "motion_change" in df.columns:
                ax1.plot(df["timestamp"], df["motion_change"], 
                        label="Motion Change", color="tab:orange", linewidth=1.5, alpha=0.7)
    elif not temporal_timestamps:
        # 如果没有frame_stats也没有异常点，显示提示信息
        ax1.text(0.5, 0.5, "No temporal frame_stats available", 
                transform=ax1.transAxes, ha="center", va="center", fontsize=12)
    
    # 标注baseline
    if temporal_baseline is not None:
        ax1.axhline(y=temporal_baseline, color="green", linestyle="--", 
                   label=f"Baseline Motion: {temporal_baseline:.2f}", linewidth=2, alpha=0.8)
    
    # 标注阈值（baseline ± threshold）
    threshold_val = motion_threshold or temporal_threshold
    if temporal_baseline is not None and threshold_val is not None:
        upper_threshold = temporal_baseline + threshold_val
        lower_threshold = temporal_baseline - threshold_val
        ax1.axhline(y=upper_threshold, color="red", linestyle="--", 
                   label=f"Baseline ± Threshold: {threshold_val:.2f}", linewidth=1.5, alpha=0.7)
        ax1.axhline(y=lower_threshold, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    
    # 标记时序变化异常
    if temporal_timestamps:
        # 优先使用motion_value，如果没有则使用motion_change
        if temporal_motion_values:
            ax1.scatter(temporal_timestamps, temporal_motion_values, 
                       color="red", s=150, marker="X", 
                       label=f"Temporal Change Anomalies ({len(temporal_change_anomalies)})", 
                       zorder=5, edgecolors="darkred", linewidths=2)
        elif temporal_motion_changes:
            ax1.scatter(temporal_timestamps, temporal_motion_changes, 
                       color="red", s=150, marker="X", 
                       label=f"Temporal Change Anomalies ({len(temporal_change_anomalies)})", 
                       zorder=5, edgecolors="darkred", linewidths=2)
        else:
            # 如果没有motion_value，使用0作为y坐标
            ax1.scatter(temporal_timestamps, [0] * len(temporal_timestamps), 
                       color="red", s=150, marker="X", 
                       label=f"Temporal Change Anomalies ({len(temporal_change_anomalies)})", 
                       zorder=5, edgecolors="darkred", linewidths=2)
    
    ax1.set_ylabel("Motion Value / Change", fontsize=12)
    ax1.set_title("Temporal Change Anomalies (Step 1: Region Temporal Analysis)", 
                 fontsize=14, fontweight="bold")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ========== 下图：运动异常 ==========
    ax2 = axes[1]
    
    # 提取运动异常的时间戳和motion_value
    motion_timestamps = []
    motion_motion_values = []
    motion_motion_changes = []
    
    if motion_anomalies:
        for anomaly in motion_anomalies:
            if isinstance(anomaly, dict):
                timestamp = anomaly.get("timestamp")
                if timestamp is not None:
                    motion_timestamps.append(float(timestamp))
                    metadata = anomaly.get("metadata", {})
                    motion_motion_values.append(metadata.get("motion_value", 0))
                    motion_motion_changes.append(metadata.get("motion_change", 0))
    
    # 绘制motion_value和motion_change曲线
    if motion_frame_stats:
        df = pd.DataFrame(motion_frame_stats)
        if "timestamp" in df.columns:
            if "motion_value" in df.columns:
                ax2.plot(df["timestamp"], df["motion_value"], 
                        label="Motion Value", color="tab:blue", linewidth=2, alpha=0.7)
            if "motion_change" in df.columns:
                ax2.plot(df["timestamp"], df["motion_change"], 
                        label="Motion Change", color="tab:orange", linewidth=1.5, alpha=0.7)
    
    # 标注baseline
    if motion_baseline is not None:
        ax2.axhline(y=motion_baseline, color="green", linestyle="--", 
                   label=f"Baseline Motion: {motion_baseline:.2f}", linewidth=2, alpha=0.8)
    
    # 标注阈值
    if motion_baseline is not None and motion_threshold is not None:
        # 标注motion_change的阈值（绝对值）
        ax2.axhline(y=motion_threshold, color="red", linestyle="--", 
                   label=f"Motion Threshold: {motion_threshold:.2f}", linewidth=1.5, alpha=0.7)
        # 标注baseline ± threshold（用于motion_value）
        upper_threshold = motion_baseline + motion_threshold
        lower_threshold = motion_baseline - motion_threshold
        ax2.axhline(y=upper_threshold, color="purple", linestyle=":", 
                   label=f"Baseline ± Threshold: {upper_threshold:.2f}", linewidth=1.5, alpha=0.7)
        ax2.axhline(y=lower_threshold, color="purple", linestyle=":", linewidth=1.5, alpha=0.7)
    
    # 标记运动异常
    if motion_timestamps:
        if motion_motion_values:
            ax2.scatter(motion_timestamps, motion_motion_values, 
                       color="red", s=150, marker="X", 
                       label=f"Motion Anomalies ({len(motion_anomalies)})", 
                       zorder=5, edgecolors="darkred", linewidths=2)
        elif motion_motion_changes:
            ax2.scatter(motion_timestamps, motion_motion_changes, 
                       color="red", s=150, marker="X", 
                       label=f"Motion Anomalies ({len(motion_anomalies)})", 
                       zorder=5, edgecolors="darkred", linewidths=2)
    
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Motion Value / Change", fontsize=12)
    ax2.set_title("Motion Anomalies (Step 2: Motion Flow Analysis)", 
                 fontsize=14, fontweight="bold")
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = (
        f"Temporal Change Anomalies: {len(temporal_change_anomalies)}\n"
        f"Motion Anomalies: {len(motion_anomalies)}\n"
    )
    if motion_baseline is not None:
        stats_text += f"\nBaseline Motion: {motion_baseline:.2f}"
    if motion_threshold is not None:
        stats_text += f"\nMotion Threshold: {motion_threshold:.2f}"
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        save_path = save_path.expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] 异常可视化已保存到: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_smoothness(
    smoothness_scores: List[float],
    timestamps: List[float],
    save_path: Optional[Path] = None,
    show: bool = True,
    title: Optional[str] = None,
    threshold: Optional[float] = None,
    anomalies: Optional[List[Dict]] = None,
    thresholds: Optional[Dict] = None,
) -> None:
    """
    绘制运动平滑度曲线
    
    Args:
        smoothness_scores: 平滑度分数列表
        timestamps: 时间戳列表
        save_path: 保存路径
        show: 是否显示
        title: 图表标题
        threshold: 可选的阈值线（手动指定）
        anomalies: 异常列表，包含frame_id和timestamp
        thresholds: 阈值字典，包含motion_threshold等
    """
    if len(smoothness_scores) != len(timestamps):
        raise ValueError("smoothness_scores 和 timestamps 长度不匹配")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # 提取异常帧的时间戳和帧ID
    anomaly_timestamps = []
    anomaly_frame_ids = []
    if anomalies:
        for anomaly in anomalies:
            if isinstance(anomaly, dict):
                # 优先使用timestamp，如果没有则使用frame_id计算
                if "timestamp" in anomaly:
                    anomaly_timestamps.append(anomaly["timestamp"])
                elif "frame_id" in anomaly:
                    # 需要根据fps计算时间戳，但这里我们使用timestamps数组
                    frame_id = anomaly["frame_id"]
                    if 0 <= frame_id < len(timestamps):
                        anomaly_timestamps.append(timestamps[frame_id])
                        anomaly_frame_ids.append(frame_id)
    
    # 1. 平滑度曲线
    axes[0].plot(timestamps, smoothness_scores, label="Motion Smoothness", color="tab:blue", linewidth=2, zorder=1)
    
    # 标记异常帧
    if anomaly_timestamps:
        # 找到异常帧对应的平滑度分数
        anomaly_scores = []
        for ts in anomaly_timestamps:
            # 找到最接近的时间戳索引
            idx = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - ts))
            if 0 <= idx < len(smoothness_scores):
                anomaly_scores.append(smoothness_scores[idx])
        
        if anomaly_scores:
            axes[0].scatter(anomaly_timestamps, anomaly_scores, 
                          color="red", s=100, marker="X", 
                          label=f"Anomalies ({len(anomaly_timestamps)})", 
                          zorder=3, edgecolors="darkred", linewidths=1.5)
    
    # 添加均值线
    axes[0].axhline(y=np.mean(smoothness_scores), color="green", linestyle="--", 
                    label=f"Mean: {np.mean(smoothness_scores):.3f}", alpha=0.7, linewidth=1.5)
    
    # 添加阈值线（如果有）
    if threshold is not None:
        axes[0].axhline(y=threshold, color="orange", linestyle="--", 
                       label=f"Custom Threshold: {threshold:.3f}", alpha=0.7, linewidth=1.5)
    
    # 添加异常检测阈值说明（注意：平滑度分数和异常检测阈值是不同的概念）
    if thresholds:
        threshold_text = "Detection Thresholds:\n"
        if "motion_threshold" in thresholds:
            threshold_text += f"Motion: {thresholds['motion_threshold']:.2f}\n"
        if "similarity_threshold" in thresholds:
            threshold_text += f"Similarity: {thresholds['similarity_threshold']:.2f}\n"
        if "hist_diff_threshold" in thresholds:
            threshold_text += f"Hist Diff: {thresholds['hist_diff_threshold']:.3f}"
        
        axes[0].text(0.98, 0.02, threshold_text, transform=axes[0].transAxes,
                    fontsize=9, verticalalignment="bottom", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    
    axes[0].set_ylabel("Smoothness Score", fontsize=12)
    axes[0].set_title("Motion Smoothness Over Time (Global Optical Flow Continuity)", fontsize=14, fontweight="bold")
    axes[0].legend(loc="best", fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.1])
    
    # 2. 平滑度分布直方图
    axes[1].hist(smoothness_scores, bins=30, alpha=0.7, color="tab:blue", edgecolor="black")
    axes[1].axvline(x=np.mean(smoothness_scores), color="green", linestyle="--", 
                   label=f"Mean: {np.mean(smoothness_scores):.3f}", linewidth=2)
    axes[1].axvline(x=np.median(smoothness_scores), color="orange", linestyle="--", 
                   label=f"Median: {np.median(smoothness_scores):.3f}", linewidth=2)
    if threshold is not None:
        axes[1].axvline(x=threshold, color="orange", linestyle="--", 
                       label=f"Custom Threshold: {threshold:.3f}", linewidth=2)
    axes[1].set_xlabel("Smoothness Score", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Smoothness Score Distribution", fontsize=14, fontweight="bold")
    axes[1].legend(loc="best", fontsize=10)
    axes[1].grid(True, alpha=0.3, axis="y")
    
    # 添加统计信息和异常信息文本
    stats_text = (
        f"Statistics:\n"
        f"Mean: {np.mean(smoothness_scores):.3f}\n"
        f"Std: {np.std(smoothness_scores):.3f}\n"
        f"Min: {np.min(smoothness_scores):.3f}\n"
        f"Max: {np.max(smoothness_scores):.3f}\n"
        f"Median: {np.median(smoothness_scores):.3f}"
    )
    
    if anomalies:
        stats_text += f"\n\nAnomalies: {len(anomalies)}"
        if anomaly_timestamps:
            stats_text += f"\nTime Range: {min(anomaly_timestamps):.2f}s - {max(anomaly_timestamps):.2f}s"
    
    # 添加说明文本
    note_text = (
        "Note: Smoothness measures global optical flow continuity.\n"
        "Anomalies are detected based on mask region analysis\n"
        "(motion change, color similarity, histogram difference)."
    )
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    fig.text(0.02, 0.98, note_text, transform=fig.transFigure,
             fontsize=8, verticalalignment="top", horizontalalignment="left",
             bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        save_path = save_path.expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] 平滑度可视化已保存到: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="可视化运动平滑度和异常检测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 可视化异常（时序变化异常和运动异常）
  python plot_motion_smoothness.py --result outputs/temporal_reasoning/video_result.json --mode anomalies
  
  # 可视化平滑度
  python plot_motion_smoothness.py --result outputs/temporal_reasoning/video_result.json --mode smoothness
  
  # 保存图像到文件
  python plot_motion_smoothness.py --result outputs/temporal_reasoning/video_result.json \\
      --mode anomalies --output outputs/anomalies_plot.png --no-show
        """
    )
    parser.add_argument("--result", required=True, help="分析结果JSON文件路径（run_analysis.py的输出）")
    parser.add_argument("--mode", choices=["anomalies", "smoothness"], default="anomalies",
                       help="可视化模式：anomalies（异常）或smoothness（平滑度）")
    parser.add_argument("--output", help="可选：保存图像的路径（png/jpg等）")
    parser.add_argument("--no-show", action="store_true", help="仅保存图像，不在窗口展示")
    parser.add_argument("--title", help="图像标题，例如视频名")
    parser.add_argument("--threshold", type=float, help="可选的阈值线（用于标记低平滑度区域，仅smoothness模式）")
    return parser.parse_args()


def main():
    args = parse_args()
    result_path = Path(args.result).expanduser().resolve()
    
    if not result_path.exists():
        raise FileNotFoundError(f"结果文件不存在: {result_path}")
    
    if args.mode == "anomalies":
        # 可视化异常
        payload = load_anomaly_data(result_path)
        
        default_title = (
            f"{Path(payload['video_path']).name} - Anomalies Analysis"
            if payload["video_path"]
            else "Anomalies Analysis"
        )
        title = args.title or default_title
        
        plot_anomalies(
            temporal_change_anomalies=payload["temporal_change_anomalies"],
            motion_anomalies=payload["motion_anomalies"],
            motion_frame_stats=payload["motion_frame_stats"],
            motion_baseline=payload["motion_baseline"],
            motion_threshold=payload["motion_threshold"],
            similarity_threshold=payload["similarity_threshold"],
            hist_diff_threshold=payload["hist_diff_threshold"],
            temporal_threshold=payload["temporal_threshold"],
            temporal_frame_stats=payload.get("temporal_frame_stats"),
            temporal_baseline=payload.get("temporal_baseline"),
            fps=payload["fps"],
            save_path=Path(args.output).expanduser().resolve() if args.output else None,
            show=not args.no_show,
            title=title,
        )
    else:
        # 可视化平滑度
        payload = load_smoothness_data(result_path)
        
        default_title = (
            f"{Path(payload['video_path']).name} - Motion Smoothness"
            if payload["video_path"]
            else "Motion Smoothness Analysis"
        )
        title = args.title or default_title
        
        plot_smoothness(
            smoothness_scores=payload["smoothness_scores"],
            timestamps=payload["timestamps"],
            save_path=Path(args.output).expanduser().resolve() if args.output else None,
            show=not args.no_show,
            title=title,
            threshold=args.threshold,
            anomalies=payload.get("anomalies", []),
            thresholds=payload.get("thresholds", {}),
        )


if __name__ == "__main__":
    main()

