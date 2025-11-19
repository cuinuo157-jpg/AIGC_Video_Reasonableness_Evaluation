# -*- coding: utf-8 -*-
"""
时序合理性评估主入口，负责统筹光流、结构、关键点及融合流程。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .config import TemporalReasoningConfig
from ..fusion.decision_engine import FusionDecisionEngine
from ..instance_tracking import TemporalCoherenceConfig, TemporalCoherencePipeline
from ..keypoint_analysis.keypoint_analyzer import KeypointAnalyzer
from ..motion_flow.flow_analyzer import MotionFlowAnalyzer


@dataclass
class StructureAnalysisOutput:
    score: float
    vanish_score: float
    emerge_score: float
    anomalies: List[Dict]
    metadata: Dict[str, object]


class TemporalReasoningAnalyzer:
    """
    时序合理性分析器
    """

    def __init__(self, config: TemporalReasoningConfig):
        """
        初始化分析器

        Args:
            config: 配置对象
        """
        self.config = config
        self.motion_analyzer: Optional[MotionFlowAnalyzer] = None
        self.structure_pipeline: Optional[TemporalCoherencePipeline] = None
        self.keypoint_analyzer: Optional[KeypointAnalyzer] = None
        self.fusion_engine: Optional[FusionDecisionEngine] = None
        self._initialized = False

    def initialize(self):
        """初始化所有子模块"""
        if self._initialized:
            print("提示：分析器已处于初始化状态。")
            return

        print("=" * 50)
        print("正在初始化时序合理性分析器...")
        print("=" * 50)

        try:
            # 1. 光流分析模块
            print("\n[1/4] 初始化光流分析器...")
            self.motion_analyzer = MotionFlowAnalyzer(self.config.raft)
            self.motion_analyzer.initialize()

            # 2. 结构一致性分析管线
            print("\n[2/4] 初始化实例追踪 / 结构分析管线...")
            coherence_config = self._build_temporal_coherence_config()
            # 将已初始化的motion_analyzer传递给structure_pipeline，避免重复初始化
            self.structure_pipeline = TemporalCoherencePipeline(coherence_config, flow_analyzer=self.motion_analyzer)
            self.structure_pipeline.initialize()

            # 3. 关键点分析器（可选，如果配置中禁用了则跳过）
            if self.config.keypoint is not None:
                print("\n[3/4] 初始化关键点分析器...")
                self.keypoint_analyzer = KeypointAnalyzer(self.config.keypoint)
                self.keypoint_analyzer.initialize()
            else:
                print("\n[3/4] 跳过关键点分析器（已禁用，通用物体检测模式）...")
                self.keypoint_analyzer = None

            # 4. 融合决策引擎
            print("\n[4/4] 初始化融合决策引擎...")
            self.fusion_engine = FusionDecisionEngine(self.config.fusion, cotracker_validator=None)

            self._initialized = True
            print("\n" + "=" * 50)
            print("时序合理性分析器初始化完成！")
            print("=" * 50)

        except Exception as exc:
            print(f"\n错误：初始化失败：{exc}")
            raise

    def _build_temporal_coherence_config(self) -> TemporalCoherenceConfig:
        """构造结构分析管线的配置字典"""
        meta_info_path = Path(self.config.output_dir) / "temporal_coherence_meta.json"
        cotracker_checkpoint = (
            self.config.tracker.cotracker_checkpoint
            or self.config.tracker.model_path
            or ".cache/scaled_offline.pth"
        )
        device = (
            "cuda"
            if "cuda" in str(self.config.device).lower() and torch.cuda.is_available()
            else "cpu"
        )

        prompts = self.config.structure_prompts or ["object"]
        prompts = [p.strip() for p in prompts if p and p.strip()]
        text_prompt = ". ".join(prompts) if prompts else "object"
        if not text_prompt.endswith("."):
            text_prompt = f"{text_prompt}."

        return TemporalCoherenceConfig(
            meta_info_path=str(meta_info_path),
            text_prompt=text_prompt,
            grounding_config_path=self.config.grounding_dino.config_path,
            grounding_checkpoint_path=self.config.grounding_dino.model_path,
            bert_path=self.config.grounding_dino.bert_path,
            sam2_config_path=self.config.sam.config_path,
            sam2_checkpoint_path=self.config.sam.model_path,
            cotracker_checkpoint_path=cotracker_checkpoint,
            device=device,
            box_threshold=self.config.grounding_dino.box_threshold,
            text_threshold=self.config.grounding_dino.text_threshold,
            grid_size=self.config.tracker.grid_size,
            iou_threshold=0.75,
            enable_visualization=self.config.structure_visualization_enable,
            visualization_output_dir=self.config.structure_visualization_output_dir,
            visualization_max_frames=self.config.structure_visualization_max_frames,
            enable_cotracker=getattr(self.config, 'enable_cotracker', False),
            cotracker_visualization_enable=self.config.cotracker_visualization_enable,
            cotracker_visualization_output_dir=self.config.cotracker_visualization_output_dir,
            cotracker_visualization_fps=self.config.cotracker_visualization_fps,
            cotracker_visualization_mode=self.config.cotracker_visualization_mode,
            cotracker_visualization_full_video=getattr(self.config, 'cotracker_visualization_full_video', False),
            enable_region_temporal_analysis=getattr(self.config, 'enable_region_temporal_analysis', True),  # 默认开启
            region_temporal_config=getattr(self.config, 'region_temporal_config', None),
        )

    def analyze(
        self,
        video_frames: List[np.ndarray],
        text_prompts: Optional[Sequence[str]] = None,
        fps: Optional[float] = None,
        video_path: Optional[str] = None,
    ) -> Dict:
        """
        分析视频时序合理性

        Args:
            video_frames: 视频帧序列，每帧为RGB图像 (H, W, 3)
            text_prompts: 文本提示列表
            fps: 视频帧率
            video_path: 原始视频路径（结构分析需要）

        Returns:
            分析结果字典
        """
        if not self._initialized:
            self.initialize()

        if not video_frames:
            raise ValueError("视频帧序列为空")

        fps = fps or 30.0

        print("\n" + "=" * 50)
        print("开始分析视频时序合理性...")
        print(f"视频帧数：{len(video_frames)}")
        print(f"视频帧率：{fps:.2f} fps")
        if text_prompts:
            print(f"文本提示：{', '.join(text_prompts)}")
        print("=" * 50)

        # 1. 结构分析（如果提供了video_path，先进行结构分析以获取masks）
        print("\n>>> 步骤1: 实例追踪 / 结构分析")
        structure_output = self._analyze_structure(video_path, text_prompts)
        
        # 从结构分析中提取已计算的光流（供步骤2复用，避免重复计算）
        optical_flows = None
        if video_path and hasattr(self.structure_pipeline, '_cached_optical_flows'):
            optical_flows = self.structure_pipeline._cached_optical_flows
            if optical_flows:
                print(f"[步骤2] 已从步骤1获取 {len(optical_flows)} 个光流场，将复用以避免重复计算")

        # 2. 光流分析（复用步骤1已计算的光流，仅计算全局平滑度）
        print("\n>>> 步骤2: 光流分析（全局平滑度）")
        if hasattr(self.config, "thresholds"):
            self.motion_analyzer.config.motion_discontinuity_threshold = (
                self.config.thresholds.motion_discontinuity_threshold
            )
        motion_score, motion_anomalies, motion_metadata = self.motion_analyzer.analyze(
            video_frames, 
            fps=fps, 
            masks=None,  # 不再使用mask，步骤2仅计算全局平滑度
            optical_flows=optical_flows  # 传入步骤1已计算的光流
        )

        # 3. 关键点分析（可选）
        if self.keypoint_analyzer is not None:
            print("\n>>> 步骤3: 关键点分析")
            physiological_score, physiological_anomalies = self.keypoint_analyzer.analyze(
                video_frames, fps=fps, video_path=video_path
            )
        else:
            print("\n>>> 步骤3: 关键点分析（已跳过）")
            physiological_score = 1.0
            physiological_anomalies = []

        # 4. 多模态融合
        print("\n>>> 步骤4: 多模态融合")
        fused_anomalies = self.fusion_engine.fuse(
            motion_anomalies,
            structure_output.anomalies,
            physiological_anomalies,
            structure_context={
                "vanish_score": structure_output.vanish_score,
                "emerge_score": structure_output.emerge_score,
                **structure_output.metadata,
            },
        )

        # 5. 计算最终得分
        print("\n>>> 步骤5: 计算最终得分")
        final_motion_score, final_structure_score = self.fusion_engine.compute_final_scores(
            motion_score,
            structure_output.score,
            physiological_score,
            fused_anomalies,
            structure_context={
                "vanish_score": structure_output.vanish_score,
                "emerge_score": structure_output.emerge_score,
                **structure_output.metadata,
            },
        )

        # 构建 sub_scores（如果关键点模块未启用，不包含 physiological_score）
        sub_scores = {
            "motion_score": float(motion_score),
            "structure_score": float(structure_output.score),
        }
        if self.keypoint_analyzer is not None:
            sub_scores["physiological_score"] = float(physiological_score)
        
        # 构建 anomaly_counts（如果关键点模块未启用，不包含 physiological）
        anomaly_counts = {
            "motion": len(motion_anomalies),
            "structure": len(structure_output.anomalies),
            "fused": len(fused_anomalies),
        }
        if self.keypoint_analyzer is not None:
            anomaly_counts["physiological"] = len(physiological_anomalies)
        
        # 收集每帧异常检测信息
        per_frame_anomaly_detection = self._collect_per_frame_anomaly_info(
            video_frames,
            motion_anomalies,
            structure_output,
            physiological_anomalies if self.keypoint_analyzer is not None else [],
            fps
        )
        
        # 收集所有阈值配置
        thresholds_info = self._collect_thresholds_info()
        
        # 只保留必要的metadata字段，避免包含大量数据（如video_object_data、frame_states等）
        structure_metadata = structure_output.metadata or {}
        filtered_structure_metadata = {
            "objects_count": structure_metadata.get("objects_count"),
            "tracking_result_length": structure_metadata.get("tracking_result_length"),
            "total_frames": structure_metadata.get("total_frames"),
            "step": structure_metadata.get("step"),
            "detection_failures": structure_metadata.get("detection_failures"),
            # 注意：不包含 video_object_data、frame_states 等大型数据，这些数据会导致JSON文件过大
        }
        # 移除 None 值
        filtered_structure_metadata = {k: v for k, v in filtered_structure_metadata.items() if v is not None}
        
        # 保存运动分析的metadata（包含frame_stats和平滑度数据，用于可视化）
        motion_metrics = {}
        if motion_metadata:
            motion_metrics = {
                "motion_threshold": motion_metadata.get("motion_threshold"),
                "similarity_threshold": motion_metadata.get("similarity_threshold"),
                "hist_diff_threshold": motion_metadata.get("hist_diff_threshold"),
                "baseline_motion": motion_metadata.get("baseline_motion"),
                "frame_stats": motion_metadata.get("frame_stats"),  # 用于可视化（mask区域）
                "smoothness_scores": motion_metadata.get("smoothness_scores"),  # 全局平滑度分数
                "smoothness_timestamps": motion_metadata.get("smoothness_timestamps"),  # 平滑度时间戳
            }
            # 移除 None 值
            motion_metrics = {k: v for k, v in motion_metrics.items() if v is not None}
        
        # 如果步骤2没有frame_stats，尝试从步骤1的区域时序分析中获取（合并所有对象）
        if not motion_metrics.get("frame_stats") and video_path and self.structure_pipeline:
            if hasattr(self.structure_pipeline, '_region_temporal_frame_stats'):
                region_frame_stats = self.structure_pipeline._region_temporal_frame_stats
                if region_frame_stats:
                    # 合并所有对象的frame_stats（按timestamp分组，取平均值或第一个）
                    import pandas as pd
                    try:
                        df = pd.DataFrame(region_frame_stats)
                        if 'timestamp' in df.columns:
                            # 按timestamp分组，取平均值（如果有多个对象在同一时间戳）
                            df_grouped = df.groupby('timestamp').agg({
                                'motion_value': 'mean',
                                'motion_change': 'mean',
                                'hist_similarity': 'mean',
                                'hist_diff': 'mean' if 'hist_diff' in df.columns else 'first',
                            }).reset_index()
                            # 转换为列表格式
                            merged_frame_stats = df_grouped.to_dict('records')
                            motion_metrics["frame_stats"] = merged_frame_stats
                        else:
                            # 如果没有timestamp，直接使用所有frame_stats
                            motion_metrics["frame_stats"] = region_frame_stats
                        
                        # 从第一个对象的结果中获取阈值和baseline信息（如果还没有）
                        # baseline_motion 在 metadata 中，不在 frame_stats 中，需要从 RegionTemporalChangeDetector 的结果中获取
                        # 但我们已经收集了 frame_stats，所以需要从其他地方获取 baseline_motion
                        # 实际上，我们需要在收集 frame_stats 时也保存 metadata
                        
                        # 从配置中获取阈值（如果还没有）
                        if self.structure_pipeline.config.region_temporal_config:
                            region_config = self.structure_pipeline.config.region_temporal_config
                            if not motion_metrics.get("motion_threshold"):
                                motion_metrics["motion_threshold"] = region_config.motion_threshold
                            if not motion_metrics.get("similarity_threshold"):
                                motion_metrics["similarity_threshold"] = region_config.similarity_threshold
                            if not motion_metrics.get("hist_diff_threshold"):
                                motion_metrics["hist_diff_threshold"] = region_config.hist_diff_threshold
                        
                        # baseline_motion 从步骤1保存的值中获取
                        if not motion_metrics.get("baseline_motion"):
                            if hasattr(self.structure_pipeline, '_region_temporal_baseline_motion'):
                                baseline = self.structure_pipeline._region_temporal_baseline_motion
                                if baseline is not None:
                                    motion_metrics["baseline_motion"] = baseline
                    except Exception as e:
                        print(f"[警告] 合并frame_stats失败: {e}，使用原始数据")
                        motion_metrics["frame_stats"] = region_frame_stats
        
        result = {
            "motion_reasonableness_score": float(final_motion_score),
            "structure_stability_score": float(final_structure_score),
            "anomalies": fused_anomalies,
            "sub_scores": sub_scores,
            "anomaly_counts": anomaly_counts,
            "structure_metrics": {
                "coherence_score": float(structure_output.score),
                "vanish_score": float(structure_output.vanish_score),
                "emerge_score": float(structure_output.emerge_score),
                **filtered_structure_metadata,
            },
            "motion_metrics": motion_metrics,  # 添加运动分析的metadata
            "per_frame_anomaly_detection": per_frame_anomaly_detection,
            "thresholds": thresholds_info,
        }

        print("\n" + "=" * 50)
        print("分析完成")
        print("=" * 50)
        print(f"运动合理性得分：{final_motion_score:.3f}")
        print(f"结构稳定性得分：{final_structure_score:.3f}")
        print(f"检测到 {len(fused_anomalies)} 个融合异常。")
        print("=" * 50)

        return result

    def _collect_per_frame_anomaly_info(
        self,
        video_frames: List[np.ndarray],
        motion_anomalies: List[Dict],
        structure_output: StructureAnalysisOutput,
        physiological_anomalies: List[Dict],
        fps: float,
    ) -> List[Dict[str, Any]]:
        """
        收集每帧的异常检测信息
        
        Returns:
            每帧的异常检测信息列表
        """
        total_frames = len(video_frames)
        per_frame_info = []
        
        # 按帧ID组织异常
        motion_by_frame = {}
        for anomaly in motion_anomalies:
            frame_id = anomaly.get('frame_id', 0)
            if frame_id not in motion_by_frame:
                motion_by_frame[frame_id] = []
            motion_by_frame[frame_id].append(anomaly)
        
        structure_by_frame = {}
        for anomaly in structure_output.anomalies:
            frame_id = anomaly.get('frame_id', 0)
            if frame_id not in structure_by_frame:
                structure_by_frame[frame_id] = []
            structure_by_frame[frame_id].append(anomaly)
        
        physiological_by_frame = {}
        for anomaly in physiological_anomalies:
            frame_id = anomaly.get('frame_id', 0)
            if frame_id not in physiological_by_frame:
                physiological_by_frame[frame_id] = []
            physiological_by_frame[frame_id].append(anomaly)
        
        # 获取结构分析的每帧信息（只保留必要的统计信息，避免包含大量数据）
        structure_frame_states = structure_output.metadata.get('frame_states', [])
        structure_frame_dict = {state.get('frame_index', -1): state for state in structure_frame_states}
        
        # 为每一帧创建信息（只包含异常和基本统计，不包含详细状态数据）
        for frame_idx in range(total_frames):
            frame_info = {
                "frame_index": frame_idx,
                "timestamp": frame_idx / max(fps, 1),
                "motion": {
                    "anomalies": motion_by_frame.get(frame_idx, []),
                    "anomaly_count": len(motion_by_frame.get(frame_idx, [])),
                },
                "structure": {
                    "anomalies": structure_by_frame.get(frame_idx, []),
                    "anomaly_count": len(structure_by_frame.get(frame_idx, [])),
                },
                "physiological": {
                    "anomalies": physiological_by_frame.get(frame_idx, []),
                    "anomaly_count": len(physiological_by_frame.get(frame_idx, [])),
                },
            }
            
            # 只添加结构分析的基本统计值（不包含完整的frame_state，避免数据过大）
            if frame_idx in structure_frame_dict:
                frame_state = structure_frame_dict[frame_idx]
                frame_info["structure"].update({
                    "object_count": frame_state.get("object_count", 0),
                    "frame_type": frame_state.get("frame_type", "unknown"),
                    "detected": frame_state.get("detected", False),
                })
            
            per_frame_info.append(frame_info)
        
        return per_frame_info
    
    def _collect_thresholds_info(self) -> Dict[str, Any]:
        """
        收集所有阈值配置信息
        
        Returns:
            阈值配置字典
        """
        thresholds_info = {}
        
        # 运动分析阈值
        if hasattr(self.config, 'thresholds'):
            thresholds_info['motion'] = {
                "motion_discontinuity_threshold": self.config.thresholds.motion_discontinuity_threshold,
            }
        
        # 结构分析阈值
        if self.structure_pipeline:
            thresholds_info['structure'] = {
                "size_change_area_ratio_threshold": self.structure_pipeline.config.size_change_area_ratio_threshold,
                "size_change_height_ratio_threshold": self.structure_pipeline.config.size_change_height_ratio_threshold,
                "size_change_min_area": self.structure_pipeline.config.size_change_min_area,
                "iou_threshold": self.structure_pipeline.config.iou_threshold,
            }
        
        # 融合阈值
        if hasattr(self.config, 'fusion'):
            thresholds_info['fusion'] = {
                "multimodal_confidence_boost": self.config.fusion.multimodal_confidence_boost,
                "min_anomaly_duration_frames": self.config.fusion.min_anomaly_duration_frames,
                "single_modality_confidence_threshold": self.config.fusion.single_modality_confidence_threshold,
            }
        
        # 关键点分析阈值
        if hasattr(self.config, 'thresholds') and hasattr(self.config.thresholds, 'keypoint_displacement_threshold'):
            thresholds_info['keypoint'] = {
                "keypoint_displacement_threshold": self.config.thresholds.keypoint_displacement_threshold,
            }
        
        # 区域时序变化检测阈值
        if self.structure_pipeline and self.structure_pipeline.config.region_temporal_config:
            region_config = self.structure_pipeline.config.region_temporal_config
            thresholds_info['region_temporal'] = {
                "motion_threshold": region_config.motion_threshold,
                "similarity_threshold": region_config.similarity_threshold,
                "consecutive_frames": region_config.consecutive_frames,
                "baseline_window": region_config.baseline_window,
                "min_roi_size": region_config.min_roi_size,
                "hist_diff_threshold": region_config.hist_diff_threshold,
            }
        
        return thresholds_info

    def _analyze_structure(
        self,
        video_path: Optional[str],
        text_prompts: Optional[Sequence[str]],
    ) -> StructureAnalysisOutput:
        if self.structure_pipeline is None:
            print("警告：结构分析管线未初始化，返回默认结果。")
            return StructureAnalysisOutput(
                score=1.0,
                vanish_score=1.0,
                emerge_score=1.0,
                anomalies=[],
                metadata={},
            )

        if not video_path:
            print("警告：未提供视频路径，无法执行结构一致性分析。")
            return StructureAnalysisOutput(
                score=1.0,
                vanish_score=1.0,
                emerge_score=1.0,
                anomalies=[],
                metadata={},
            )

        try:
            result = self.structure_pipeline.evaluate_video(video_path, text_prompts)
            return StructureAnalysisOutput(
                score=float(result.coherence_score),
                vanish_score=float(result.vanish_score),
                emerge_score=float(result.emerge_score),
                anomalies=result.anomalies,
                metadata=result.metadata,
            )
        except Exception as exc:
            print(f"警告：结构分析失败，将使用默认得分。详情：{exc}")
            return StructureAnalysisOutput(
                score=1.0,
                vanish_score=1.0,
                emerge_score=1.0,
                anomalies=[],
                metadata={"error": str(exc)},
            )
    
    def _extract_combined_masks_from_structure(
        self,
        structure_metadata: Dict[str, object],
        num_frames: int
    ) -> Optional[List[Optional[np.ndarray]]]:
        """
        从结构分析的metadata中提取合并后的masks
        
        Args:
            structure_metadata: 结构分析的metadata字典
            num_frames: 视频帧数
        
        Returns:
            masks列表，每帧一个mask（合并所有对象的mask），如果无法提取则返回None
        """
        # 尝试从metadata中获取video_object_data
        video_object_data = structure_metadata.get('video_object_data')
        if not video_object_data or not isinstance(video_object_data, list):
            return None
        
        # 确保长度匹配
        min_len = min(len(video_object_data), num_frames)
        combined_masks = []
        
        for frame_idx in range(min_len):
            frame_data = video_object_data[frame_idx]
            if not isinstance(frame_data, dict) or not frame_data:
                combined_masks.append(None)
                continue
            
            # 合并该帧所有对象的mask
            frame_mask = None
            for obj_id, obj_info in frame_data.items():
                if not isinstance(obj_info, dict):
                    continue
                
                mask_data = obj_info.get('mask')
                if mask_data is None:
                    continue
                
                # 转换mask为numpy数组
                if isinstance(mask_data, list):
                    mask_np = np.array(mask_data, dtype=bool)
                elif isinstance(mask_data, torch.Tensor):
                    mask_np = mask_data.cpu().numpy().astype(bool)
                else:
                    mask_np = np.array(mask_data, dtype=bool)
                
                # 确保mask是2D
                if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                    mask_np = mask_np[0]
                elif mask_np.ndim == 3:
                    mask_np = mask_np[:, :, 0] if mask_np.shape[2] == 1 else mask_np
                
                # 合并到frame_mask
                if frame_mask is None:
                    frame_mask = mask_np.astype(bool)
                else:
                    # 确保形状一致
                    if frame_mask.shape == mask_np.shape:
                        frame_mask = frame_mask | mask_np.astype(bool)
            
            combined_masks.append(frame_mask)
        
        # 如果所有帧都没有mask，返回None
        if all(m is None for m in combined_masks):
            return None
        
        return combined_masks

