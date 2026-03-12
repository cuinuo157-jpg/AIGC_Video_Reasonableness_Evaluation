# -*- coding: utf-8 -*-
"""
视频模糊检测可视化工具
提供模糊检测过程的图表展示与报告生成功能。
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch


class BlurVisualization:
    """模糊检测结果的可视化辅助类。"""
    
    def __init__(self, output_dir: str = "./blur_visualization_results"):
        """
        初始化可视化工具。
        
        Args:
            output_dir: 可视化输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置 matplotlib 字体（默认英文字体，不出现乱码）
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置 seaborn 样式
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def visualize_quality_scores(
        self,
        video_path: str,
        quality_scores: List[float],
        blur_frames: List[int],
        threshold: float,
        save_path: Optional[str] = None
    ) -> str:
        """
        可视化质量分数随帧变化的曲线与分布。
        
        Args:
            video_path: 视频路径
            quality_scores: 质量分数列表
            blur_frames: 检测到的模糊帧索引
            threshold: 检测阈值
            save_path: 保存路径，默认输出目录
            
        Returns:
            生成图像的保存路径
        """
        if save_path is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            save_path = str(self.output_dir / f"{video_name}_quality_scores.png")
        
        # 绘制图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 质量分数折线图
        frame_indices = list(range(len(quality_scores)))
        ax1.plot(
            frame_indices, quality_scores, 'b-', linewidth=2,
            label='Quality Score'
        )
        ax1.axhline(
            y=threshold, color='r', linestyle='--', linewidth=2,
            label=f'Detection Threshold ({threshold:.3f})'
        )
        
        # ���ģ��֡
        if blur_frames:
            blur_scores = [quality_scores[i] for i in blur_frames if i < len(quality_scores)]
            valid_blur_frames = [i for i in blur_frames if i < len(quality_scores)]
            if blur_scores:
                ax1.scatter(
                    valid_blur_frames, blur_scores, color='red', s=50,
                    zorder=5, label=f'Blur Frames ({len(blur_frames)})'
                )
        
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Quality Score')
        ax1.set_title(
            f'Video Quality Score Changes - {os.path.basename(video_path)}'
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 质量分数直方图
        ax2.hist(
            quality_scores, bins=30, alpha=0.7, color='skyblue',
            edgecolor='black'
        )
        ax2.axvline(
            x=threshold, color='r', linestyle='--', linewidth=2,
            label=f'Detection Threshold ({threshold:.3f})'
        )
        mean_quality = np.mean(quality_scores)
        ax2.axvline(
            x=mean_quality, color='g', linestyle='-', linewidth=2,
            label=f'Mean Quality ({mean_quality:.3f})'
        )
        
        ax2.set_xlabel('Quality Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Quality Score Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_blur_frames(
        self,
        video_path: str,
        blur_frames: List[int],
        num_samples: int = 6,
        save_path: Optional[str] = None
    ) -> str:
        """
        可视化部分模糊帧的图像内容。
        
        Args:
            video_path: 视频路径
            blur_frames: 模糊帧索引列表
            num_samples: 采样展示数量
            save_path: 保存路径
            
        Returns:
            生成图像的保存路径
        """
        if save_path is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            save_path = str(self.output_dir / f"{video_name}_blur_frames.png")
        
        # 读取视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法读取视频: {video_path}")
            return ""
        
        frames = []
        # 读取全部帧
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release()
        
        if not frames:
            print("无法获取视频帧")
            return ""
        
        # 筛选需要展示的模糊帧
        if len(blur_frames) > num_samples:
            selected_indices = np.linspace(
                0, len(blur_frames) - 1, num_samples, dtype=int
            )
            selected_blur_frames = [blur_frames[i] for i in selected_indices]
        else:
            selected_blur_frames = blur_frames
        
        # 绘制展示图
        cols = 3
        rows = (len(selected_blur_frames) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, frame_idx in enumerate(selected_blur_frames):
            row = i // cols
            col = i % cols
            
            if frame_idx < len(frames):
                axes[row, col].imshow(frames[frame_idx])
                axes[row, col].set_title(f'Blur Frame {frame_idx}')
                axes[row, col].axis('off')
            else:
                axes[row, col].axis('off')
        
        # 关掉无需使用的子图
        for i in range(len(selected_blur_frames), rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.suptitle(
            f'Detected Blur Frames - {os.path.basename(video_path)}',
            fontsize=16
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_batch_results(
        self,
        results: List[Dict],
        save_path: Optional[str] = None
    ) -> str:
        """
        对批量检测结果进行可视化分析。
        
        Args:
            results: 检测结果列表
            save_path: 保存路径
            
        Returns:
            生成图像的保存路径
        """
        if save_path is None:
            save_path = str(
                self.output_dir / "batch_results_visualization.png"
            )
        
        # 提取数据
        video_names = [
            os.path.basename(r.get('video_path', '')) for r in results
        ]
        confidences = [r.get('confidence', 0.0) for r in results]
        blur_detected = [r.get('blur_detected', False) for r in results]
        blur_ratios = [r.get('blur_ratio', 0.0) for r in results]
        
        # 绘制图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. 置信度分布
        ax1.hist(
            confidences, bins=20, alpha=0.7, color='skyblue',
            edgecolor='black'
        )
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Confidence Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 2. 模糊比例分布
        ax2.hist(
            blur_ratios, bins=20, alpha=0.7, color='lightcoral',
            edgecolor='black'
        )
        ax2.set_xlabel('Blur Ratio')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Blur Ratio Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. 置信度 vs 模糊比例散点图
        scatter_colors = ['red' if detected else 'blue' for detected in blur_detected]
        ax3.scatter(
            confidences, blur_ratios, c=scatter_colors, alpha=0.6, s=50
        )
        ax3.set_xlabel('Confidence')
        ax3.set_ylabel('Blur Ratio')
        ax3.set_title('Confidence vs Blur Ratio')
        ax3.grid(True, alpha=0.3)
        
        # 散点图图例
        legend_elements = [
            Patch(facecolor='red', label='Blur Detected'),
            Patch(facecolor='blue', label='No Blur Detected')
        ]
        ax3.legend(handles=legend_elements)
        
        # 4. 检测结果统计饼图
        blur_count = sum(blur_detected)
        no_blur_count = len(blur_detected) - blur_count
        
        labels = ['Blur Detected', 'No Blur Detected']
        sizes = [blur_count, no_blur_count]
        pie_colors = ['lightcoral', 'lightblue']
        
        ax4.pie(
            sizes, labels=labels, colors=pie_colors,
            autopct='%1.1f%%', startangle=90
        )
        ax4.set_title('Detection Results Statistics')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_detection_report(
        self,
        result: Dict,
        save_path: Optional[str] = None
    ) -> str:
        """
        生成单个视频的模糊检测报告图。
        
        Args:
            result: 检测结果字典
            save_path: 保存路径
            
        Returns:
            生成图像的保存路径
        """
        if save_path is None:
            video_name = os.path.splitext(
                os.path.basename(result.get('video_path', ''))
            )[0]
            save_path = str(
                self.output_dir / f"{video_name}_detection_report.png"
            )
        
        # 创建画布
        fig = plt.figure(figsize=(16, 12))
        
        # 总标题
        fig.suptitle(
            f'Video Blur Detection Report - {os.path.basename(result.get("video_path", ""))}',
            fontsize=20, fontweight='bold'
        )
        
        # 定义子图布局
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 结果概览
        ax1 = fig.add_subplot(gs[0, 0])
        blur_detected = result.get('blur_detected', False)
        confidence = result.get('confidence', 0.0)
        
        # 结果指示
        result_color = 'red' if blur_detected else 'green'
        result_text = "Blur Detected" if blur_detected else "No Blur Detected"
        ax1.text(
            0.5, 0.5, f'Result: {result_text}',
            ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor=result_color, alpha=0.3
            )
        )
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # 2. 置信度
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(['Confidence'], [confidence], color='skyblue', alpha=0.7)
        ax2.set_ylabel('Confidence')
        ax2.set_ylim(0, 1)
        ax2.set_title('Detection Confidence')
        
        # 3. 模糊严重程度
        ax3 = fig.add_subplot(gs[0, 2])
        severity = result.get('blur_severity', 'Unknown')
        severity_colors = {
            '无模糊': 'green',
            '轻微模糊': 'yellow',
            '中等模糊': 'orange',
            '严重模糊': 'red'
        }
        severity_color = severity_colors.get(severity, 'gray')
        
        ax3.text(
            0.5, 0.5, f'Severity:\n{severity}',
            ha='center', va='center', fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor=severity_color, alpha=0.3
            )
        )
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        
        # 4. 关键指标
        ax4 = fig.add_subplot(gs[1, :])
        metrics = {
            'Blur Ratio': result.get('blur_ratio', 0.0),
            'Blur Frames': result.get('blur_frame_count', 0),
            'Total Frames': result.get('total_frames', 0),
            'Avg Quality': result.get('avg_quality', 0.0),
            'Max Quality Drop': result.get('max_quality_drop', 0.0)
        }
        
        bars = ax4.bar(
            metrics.keys(), metrics.values(), color='lightblue', alpha=0.7
        )
        ax4.set_ylabel('Value')
        ax4.set_title('Key Detection Metrics')
        ax4.tick_params(axis='x', rotation=45)
        
        # ������ֵ��ǩ
        for bar, value in zip(bars, metrics.values()):
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{value:.3f}' if isinstance(value, float) else f'{value}',
                ha='center', va='bottom'
            )
        
        # 5. 建议
        ax5 = fig.add_subplot(gs[2, :])
        recommendations = result.get('recommendations', [])
        if recommendations:
            rec_text = '\n'.join([f'• {rec}' for rec in recommendations])
        else:
            rec_text = 'No specific recommendations'
        
        ax5.text(
            0.05, 0.95, f'Recommendations:\n{rec_text}',
            transform=ax5.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8
            )
        )
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

