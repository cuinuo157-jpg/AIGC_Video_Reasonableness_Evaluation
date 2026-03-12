# -*- coding: utf-8 -*-
"""
视频模糊检测器
基于 Q-Align 模型的运动平滑度评分，用于快速检测视频模糊
"""

import os
import csv
import json
import warnings
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image

# 忽略警告
warnings.filterwarnings("ignore")

# 导入本地模块
from motion_smoothness_score import (
    QAlignVideoScorer,
    load_video_with_sliding_window,
    load_video_with_sliding_window_generator,
    calculate_adaptive_threshold,
    detect_artifact_frames
)


class BlurDetector:
    """视频模糊检测器，基于 Q-Align 模型的运动平滑度评分"""
    
    # 模糊严重程度阈值定义
    BLUR_THRESHOLDS = {
        'mild': 0.015,      # 轻微模糊阈值
        'moderate': 0.025,  # 中等模糊阈值
        'severe': 0.04      # 严重模糊阈值
    }
    
    # 模糊比例阈值
    BLUR_RATIO_THRESHOLD = 0.05
    
    def __init__(
        self,
        device: str = "cuda:0",
        model_path: str = ".cache/q-future/one-align",
        batch_size: int = 32
    ):
        """
        初始化模糊检测器
        
        Args:
            device: 计算设备 (cuda:0, cpu 等)
            model_path: Q-Align 模型路径
            batch_size: 批处理大小，用于控制内存使用。默认 32
        """
        self.device = device
        self.model_path = model_path
        self.batch_size = batch_size
        
        # 初始化 Q-Align 模型
        print("正在初始化 Q-Align 模型...")
        try:
            self.scorer = QAlignVideoScorer(pretrained=model_path, device=device)
            print("Q-Align 模型初始化完成！")
        except Exception as e:
            print(f"模型初始化失败: {e}")
            raise
    
    def detect_blur(
        self,
        video_path: str,
        window_size: int = 3,
        batch_size: Optional[int] = None
    ) -> Dict:
        """
        检测视频模糊
        
        Args:
            video_path: 视频文件路径
            window_size: 滑动窗口大小（帧数）
            batch_size: 批处理大小，用于控制内存使用。默认 32，可根据 GPU 内存调整
            
        Returns:
            模糊检测结果字典，包含：
            - video_path: 视频路径
            - blur_detected: 是否检测到模糊
            - blur_severity: 模糊严重程度
            - confidence: 检测置信度
            - blur_ratio: 模糊帧比例
            - 其他相关指标
        """
        video_name = os.path.basename(video_path)
        print(f"开始检测视频模糊: {video_name}")
        
        try:
            # 1. 加载视频帧
            video_frames = load_video_with_sliding_window(video_path, window_size)
            
            # 2. 计算质量分数（使用批处理）
            # 如果没有指定 batch_size，使用实例的默认 batch_size
            effective_batch_size = batch_size if batch_size is not None else self.batch_size
            total_frames = len(video_frames)
            print(f"使用批处理大小: {effective_batch_size}，处理 {total_frames} 个帧组...")
            
            _, _, quality_scores = self.scorer(video_frames, batch_size=effective_batch_size)
            quality_scores = quality_scores.tolist()
            
            # 释放视频帧内存
            del video_frames
            
            # 3. 估算相机运动幅度
            camera_movement = self._estimate_camera_movement(video_path)
            
            # 4. 计算自适应阈值
            threshold = calculate_adaptive_threshold(camera_movement)
            
            # 5. 检测模糊帧
            blur_frame_indices = detect_artifact_frames(quality_scores, threshold)
            
            # 6. 计算模糊指标
            blur_metrics = self._calculate_blur_metrics(
                quality_scores, blur_frame_indices, threshold
            )
            
            # 7. 生成检测结果
            result = self._generate_detection_result(video_path, blur_metrics)
            
            return result
            
        except Exception as e:
            print(f"模糊检测失败: {e}")
            return {
                'video_path': video_path,
                'video_name': video_name,
                'blur_detected': False,
                'confidence': 0.0,
                'error': str(e),
                'blur_severity': '检测失败'
            }
    
    def _estimate_camera_movement(self, video_path: str, sample_frames: int = 10) -> float:
        """
        估算相机运动幅度
        
        Args:
            video_path: 视频文件路径
            sample_frames: 采样帧数
            
        Returns:
            相机运动幅度 (0-1之间)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0.0
            
            frames = []
            frame_count = 0
            
            # 读取前 N 帧用于估算运动
            while frame_count < sample_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray_frame)
                frame_count += 1
            
            cap.release()
            
            if len(frames) < 2:
                return 0.0
            
            # 计算相邻帧之间的差异
            total_diff = 0.0
            for i in range(1, len(frames)):
                diff = cv2.absdiff(frames[i], frames[i-1])
                total_diff += np.mean(diff)
            
            # 归一化运动幅度到 [0, 1]
            movement = total_diff / (len(frames) - 1) / 255.0
            return min(1.0, max(0.0, movement))
            
        except Exception as e:
            print(f"相机运动估算失败: {e}")
            return 0.0
    
    def _calculate_blur_metrics(
        self,
        quality_scores: List[float],
        blur_frame_indices: np.ndarray,
        threshold: float
    ) -> Dict:
        """
        计算模糊相关指标
        
        Args:
            quality_scores: 质量分数列表
            blur_frame_indices: 模糊帧索引数组
            threshold: 检测阈值
            
        Returns:
            模糊指标字典
        """
        total_frames = len(quality_scores)
        blur_frame_count = len(blur_frame_indices)
        
        # 基础指标
        blur_ratio = blur_frame_count / total_frames if total_frames > 0 else 0.0
        avg_quality = float(np.mean(quality_scores))
        quality_std = float(np.std(quality_scores))
        
        # 计算质量分数变化
        quality_diffs = np.abs(np.diff(quality_scores))
        max_quality_drop = float(np.max(quality_diffs)) if len(quality_diffs) > 0 else 0.0
        
        # 确定模糊严重程度
        blur_severity = self._determine_blur_severity(blur_ratio, max_quality_drop, threshold)
        
        # 计算综合置信度
        confidence = self._calculate_confidence(blur_ratio, max_quality_drop, avg_quality)
        
        # 判断是否检测到模糊
        blur_detected = (
            blur_ratio > self.BLUR_RATIO_THRESHOLD or
            max_quality_drop > threshold
        )
        
        return {
            'total_frames': total_frames,
            'blur_frames': blur_frame_indices.tolist(),
            'blur_frame_count': blur_frame_count,
            'blur_ratio': blur_ratio,
            'avg_quality': avg_quality,
            'quality_std': quality_std,
            'max_quality_drop': max_quality_drop,
            'threshold': threshold,
            'blur_severity': blur_severity,
            'confidence': confidence,
            'blur_detected': blur_detected,
            'quality_scores': quality_scores
        }
    
    def _determine_blur_severity(
        self,
        blur_ratio: float,
        max_quality_drop: float,
        threshold: float
    ) -> str:
        """
        确定模糊严重程度
        
        Args:
            blur_ratio: 模糊帧比例
            max_quality_drop: 最大质量下降
            threshold: 检测阈值
            
        Returns:
            模糊严重程度字符串
        """
        if blur_ratio > 0.3 or max_quality_drop > threshold * 2:
            return "严重模糊"
        elif blur_ratio > 0.1 or max_quality_drop > threshold * 1.5:
            return "中等模糊"
        elif blur_ratio > self.BLUR_RATIO_THRESHOLD or max_quality_drop > threshold:
            return "轻微模糊"
        else:
            return "无模糊"
    
    def _calculate_confidence(
        self,
        blur_ratio: float,
        max_quality_drop: float,
        avg_quality: float
    ) -> float:
        """
        计算模糊检测置信度
        
        Args:
            blur_ratio: 模糊帧比例
            max_quality_drop: 最大质量下降
            avg_quality: 平均质量分数
            
        Returns:
            置信度 (0-1之间)
        """
        # 基于模糊比例和质量下降计算置信度
        blur_confidence = min(1.0, blur_ratio * 2.0)  # 模糊比例权重
        quality_confidence = min(1.0, max_quality_drop * 10.0)  # 质量下降权重
        avg_quality_confidence = max(0.0, 1.0 - avg_quality)  # 平均质量权重
        
        # 加权综合置信度
        confidence = (
            blur_confidence * 0.4 +
            quality_confidence * 0.4 +
            avg_quality_confidence * 0.2
        )
        
        return min(1.0, max(0.0, confidence))
    
    def _generate_detection_result(self, video_path: str, blur_metrics: Dict) -> Dict:
        """
        生成检测结果字典
        
        Args:
            video_path: 视频文件路径
            blur_metrics: 模糊指标字典
            
        Returns:
            完整的检测结果字典
        """
        return {
            'video_path': video_path,
            'video_name': os.path.basename(video_path),
            'blur_detected': blur_metrics['blur_detected'],
            'confidence': blur_metrics['confidence'],
            'blur_severity': blur_metrics['blur_severity'],
            'blur_ratio': blur_metrics['blur_ratio'],
            'blur_frame_count': blur_metrics['blur_frame_count'],
            'total_frames': blur_metrics['total_frames'],
            'avg_quality': blur_metrics['avg_quality'],
            'quality_std': blur_metrics['quality_std'],
            'max_quality_drop': blur_metrics['max_quality_drop'],
            'threshold': blur_metrics['threshold'],
            'blur_frames': blur_metrics['blur_frames'],
            'quality_scores': blur_metrics.get('quality_scores', []),
            'recommendations': self._generate_recommendations(blur_metrics)
        }
    
    def _generate_recommendations(self, blur_metrics: Dict) -> List[str]:
        """
        根据检测结果生成改进建议
        
        Args:
            blur_metrics: 模糊指标字典
            
        Returns:
            建议列表
        """
        recommendations = []
        
        if not blur_metrics['blur_detected']:
            recommendations.append("视频质量良好")
            recommendations.append("无需特殊处理")
            return recommendations
        
        severity = blur_metrics['blur_severity']
        
        if severity == "严重模糊":
            recommendations.append("建议重新录制视频")
            recommendations.append("使用三脚架或稳定器")
            recommendations.append("检查相机对焦设置")
            recommendations.append("提高录制分辨率")
        elif severity == "中等模糊":
            recommendations.append("建议使用稳定器")
            recommendations.append("提高录制帧率")
            recommendations.append("确保充足光线")
            recommendations.append("可考虑后期降噪处理")
        else:  # 轻微模糊
            recommendations.append("可考虑后期处理")
            recommendations.append("轻微模糊，影响较小")
            recommendations.append("可选择性修复关键帧")
        
        return recommendations
    
    def batch_detect(
        self,
        video_dir: str,
        output_dir: str = "./blur_detection_results",
        video_extensions: List[str] = None
    ) -> Dict:
        """
        批量检测视频模糊
        
        Args:
            video_dir: 视频目录路径
            output_dir: 输出目录路径
            video_extensions: 视频文件扩展名列表，默认为 ['.mp4', '.avi', '.mov', '.mkv']
            
        Returns:
            批量检测结果字典
        """
        if video_extensions is None:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有视频文件
        video_files = []
        video_dir_path = Path(video_dir)
        for ext in video_extensions:
            video_files.extend(video_dir_path.glob(f'*{ext}'))
            video_files.extend(video_dir_path.glob(f'*{ext.upper()}'))
        
        if not video_files:
            print(f"在目录 {video_dir} 中未找到视频文件")
            return {
                'total_videos': 0,
                'processed_videos': 0,
                'blur_detected_count': 0,
                'results': []
            }
        
        results = []
        print(f"开始批量检测 {len(video_files)} 个视频...")
        
        for video_file in video_files:
            try:
                result = self.detect_blur(str(video_file))
                results.append(result)
                print(
                    f"  {video_file.name}: {result['blur_severity']} "
                    f"(置信度: {result['confidence']:.3f})"
                )
            except Exception as e:
                print(f"  处理 {video_file.name} 时出错: {e}")
                results.append({
                    'video_path': str(video_file),
                    'video_name': video_file.name,
                    'blur_detected': False,
                    'confidence': 0.0,
                    'error': str(e),
                    'blur_severity': '检测失败'
                })
        
        # 保存结果
        self._save_results(results, output_dir)
        
        blur_detected_count = sum(
            1 for r in results if r.get('blur_detected', False)
        )
        
        return {
            'total_videos': len(video_files),
            'processed_videos': len(results),
            'blur_detected_count': blur_detected_count,
            'results': results
        }
    
    @staticmethod
    def _make_json_serializable(obj):
        """
        将 NumPy/PyTorch 类型转换为 Python 原生类型
        
        Args:
            obj: 待转换的对象
            
        Returns:
            可序列化的 Python 对象
        """
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [
                BlurDetector._make_json_serializable(item) for item in obj
            ]
        elif isinstance(obj, dict):
            return {
                key: BlurDetector._make_json_serializable(value)
                for key, value in obj.items()
            }
        elif hasattr(obj, 'item'):  # PyTorch tensor
            return obj.item()
        else:
            return obj
    
    def _save_results(self, results: List[Dict], output_dir: str):
        """
        保存检测结果到文件
        
        Args:
            results: 检测结果列表
            output_dir: 输出目录路径
        """
        # 保存 JSON 结果
        json_path = os.path.join(output_dir, 'blur_detection_results.json')
        serializable_results = self._make_json_serializable(results)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # 保存 CSV 摘要
        csv_path = os.path.join(output_dir, 'blur_detection_summary.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Video', 'Blur_Detected', 'Confidence', 'Severity',
                'Blur_Ratio', 'Blur_Frames', 'Total_Frames', 'Avg_Quality'
            ])
            
            for result in results:
                writer.writerow([
                    result.get('video_name', os.path.basename(
                        result.get('video_path', '')
                    )),
                    result.get('blur_detected', False),
                    f"{result.get('confidence', 0.0):.3f}",
                    result.get('blur_severity', ''),
                    f"{result.get('blur_ratio', 0.0):.3f}",
                    result.get('blur_frame_count', 0),
                    result.get('total_frames', 0),
                    f"{result.get('avg_quality', 0.0):.3f}"
                ])
        
        # 生成统计报告
        self._generate_statistics_report(results, output_dir)
        
        print(f"检测结果已保存到: {output_dir}")
    
    def _generate_statistics_report(self, results: List[Dict], output_dir: str):
        """
        生成统计报告
        
        Args:
            results: 检测结果列表
            output_dir: 输出目录路径
        """
        total_videos = len(results)
        if total_videos == 0:
            return
        
        # 计算基本统计信息
        blur_detected_count = sum(
            1 for r in results if r.get('blur_detected', False)
        )
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return
        
        confidence_scores = [r.get('confidence', 0.0) for r in valid_results]
        blur_ratios = [r.get('blur_ratio', 0.0) for r in valid_results]
        
        # 统计模糊严重程度分布
        severity_counts = {}
        for result in results:
            if result.get('blur_detected', False):
                severity = result.get('blur_severity', '未知')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # 生成报告
        report = f"""# 视频模糊检测统计报告

## 基本统计
- 总视频数量: {total_videos}
- 检测到模糊的视频: {blur_detected_count}
- 模糊检测率: {blur_detected_count/total_videos*100:.1f}%

## 置信度统计
- 平均置信度: {np.mean(confidence_scores):.3f}
- 最低置信度: {np.min(confidence_scores):.3f}
- 最高置信度: {np.max(confidence_scores):.3f}
- 置信度标准差: {np.std(confidence_scores):.3f}

## 模糊比例统计
- 平均模糊比例: {np.mean(blur_ratios):.3f}
- 最低模糊比例: {np.min(blur_ratios):.3f}
- 最高模糊比例: {np.max(blur_ratios):.3f}

## 模糊严重程度分布
"""
        
        if severity_counts:
            for severity, count in sorted(severity_counts.items()):
                report += f"- {severity}: {count} 个视频 ({count/total_videos*100:.1f}%)\n"
        else:
            report += "- 无模糊检测\n"
        
        # 保存报告
        report_path = os.path.join(output_dir, 'statistics_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
