# -*- coding: utf-8 -*-
"""
时序合理性主流程的配置定义。
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class RAFTConfig:
    """RAFT 光流模型配置。"""
    model_path: str = ""
    model_type: str = "large"  # 支持 large 或 small
    use_gpu: bool = True
    batch_size: int = 1
    enable_visualization: bool = False
    visualization_output_dir: Optional[str] = None
    visualization_max_frames: int = 30


@dataclass
class GroundingDINOConfig:
    """Grounding DINO 检测模型配置。"""
    model_path: str = ""  # 模型权重路径
    config_path: str = ""  # 配置文件路径
    bert_path: str = ""  # 本地 BERT 模型目录
    text_threshold: float = 0.25
    box_threshold: float = 0.3
    use_gpu: bool = True


@dataclass
class SAMConfig:
    """SAM / SAM2 分割模型配置。"""
    model_path: str = ""  # 权重文件路径
    config_path: str = ""  # 配置文件路径（SAM2 必填项）
    model_type: str = "sam2_h"  # 可选 sam2_h / sam2_l / sam2_b
    use_gpu: bool = True
    resolved_config_path: Optional[str] = field(default=None, init=False)


@dataclass
class TrackerConfig:
    """跟踪 / 验证模块配置。"""
    type: str = "deaot"  # 支持 deaot 或 cotracker
    model_path: Optional[str] = None
    use_gpu: bool = True
    enable_cotracker_validation: bool = True  # 是否启用 Co-Tracker 校验
    cotracker_checkpoint: Optional[str] = None  # Co-Tracker 权重路径
    grid_size: int = 30  # Co-Tracker 网格密度


@dataclass
class KeypointConfig:
    """关键点分析配置。"""
    model_type: str = "mediapipe"  # 支持 mediapipe 或 mmpose
    model_path: Optional[str] = None
    use_gpu: bool = False  # MediaPipe 默认仅支持 CPU
    
    # 可选可视化配置
    enable_visualization: bool = False  # 是否开启关键点可视化
    visualization_output_dir: Optional[str] = None  # 可视化结果输出目录
    show_face: bool = False  # 是否绘制人脸关键点
    show_face_mesh: bool = False  # 是否绘制人脸网格
    point_radius: int = 3  # 关键点绘制半径
    line_thickness: int = 2  # 关键点连线粗细
    save_visualization: bool = True  # 是否保存可视化结果
    show_visualization: bool = False  # 是否弹窗展示可视化


@dataclass
class FusionConfig:
    """多模态融合与打分配置。"""
    multimodal_confidence_boost: float = 1.2
    min_anomaly_duration_frames: int = 1  # 从3降到1，允许单帧异常
    single_modality_confidence_threshold: float = 0.5  # 从0.8降到0.5


@dataclass
class ThresholdsConfig:
    """通用阈值配置。"""
    motion_discontinuity_threshold: float = 0.3
    structure_disappearance_threshold: float = 0.3  # 结构消失判断阈值
    keypoint_displacement_threshold: int = 10  # 关键点位移阈值（像素）


@dataclass
class TemporalReasoningConfig:
    """时序合理性主流程配置。"""
    device: str = "cuda:0"
    
    # 子模块配置
    raft: RAFTConfig = field(default_factory=RAFTConfig)
    grounding_dino: GroundingDINOConfig = field(default_factory=GroundingDINOConfig)
    sam: SAMConfig = field(default_factory=SAMConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    keypoint: Optional[KeypointConfig] = field(default_factory=KeypointConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    structure_prompts: Optional[List[str]] = None
    structure_visualization_enable: bool = False
    structure_visualization_output_dir: Optional[str] = None
    structure_visualization_max_frames: int = 50
    enable_cotracker: bool = False  # 是否启用CoTracker验证（默认不启用，仅使用SAM2评估）
    cotracker_visualization_enable: bool = False
    cotracker_visualization_output_dir: Optional[str] = None
    cotracker_visualization_fps: int = 12
    cotracker_visualization_mode: str = "rainbow"
    cotracker_visualization_full_video: bool = False  # 是否生成整体视频的追踪可视化（默认不生成）
    enable_region_temporal_analysis: bool = True  # 是否启用区域时序变化检测（默认开启，使用SAM2的mask进行光流和颜色相似度检测）
    region_temporal_config: Optional[Any] = None  # 区域时序变化检测配置（RegionTemporalChangeConfig）
    
    # 输出配置
    output_dir: str = ""
    save_visualizations: bool = True
    save_detailed_reports: bool = True
    
    @staticmethod
    def _resolve_path(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        return str(Path(path).expanduser().resolve())
    
    def __post_init__(self):
        """初始化默认路径与依赖关系。"""
        # 推导默认目录结构
        base_dir = Path(__file__).parent.parent.parent.parent
        third_party_dir = base_dir / "third_party"
        cache_dir = base_dir / ".cache"
        outputs_dir = base_dir / "outputs" / "temporal_reasoning"
        cotracker_vis_dir = base_dir / "outputs" / "cotracker_visualization"
        
        # RAFT 默认路径（权重：.cache，代码：third_party）
        if self.raft.model_path:
            self.raft.model_path = self._resolve_path(self.raft.model_path)
        else:
            raft_cache_path = (cache_dir / "raft-things.pth").resolve()
            if raft_cache_path.exists():
                self.raft.model_path = str(raft_cache_path)
            else:
                raise FileNotFoundError(
                    f"未找到 RAFT 模型权重：{raft_cache_path}\n"
                    f"请确认权重位于 .cache 目录。"
                )
        
        # Grounding DINO 默认路径
        if self.grounding_dino.model_path:
            self.grounding_dino.model_path = self._resolve_path(self.grounding_dino.model_path)
        else:
            gdino_weight = (cache_dir / "groundingdino_swinb_cogcoor.pth").resolve()
            if gdino_weight.exists():
                self.grounding_dino.model_path = str(gdino_weight)
            else:
                raise FileNotFoundError(
                    f"未找到 Grounding DINO 权重：{gdino_weight}\n"
                    f"请确认权重位于 .cache 目录。"
                )
        
        # Grounding DINO 配置文件
        if self.grounding_dino.config_path:
            self.grounding_dino.config_path = self._resolve_path(self.grounding_dino.config_path)
        else:
            gdino_config = (third_party_dir / "Grounded-SAM-2" / "grounding_dino" / "config" / "GroundingDINO_SwinB.py").resolve()
            if not gdino_config.exists():
                gdino_config = (third_party_dir / "Grounded-Segment-Anything" / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinB.py").resolve()
            if gdino_config.exists():
                self.grounding_dino.config_path = str(gdino_config)
            else:
                raise FileNotFoundError(
                    f"未找到 Grounding DINO 配置文件：{gdino_config}\n"
                    f"请确认配置文件存在于 third_party 目录。"
                )
        
        # BERT 模型路径
        if self.grounding_dino.bert_path:
            self.grounding_dino.bert_path = self._resolve_path(self.grounding_dino.bert_path)
        else:
            bert_path = (cache_dir / "google-bert" / "bert-base-uncased").resolve()
            if bert_path.exists():
                self.grounding_dino.bert_path = str(bert_path)
            else:
                raise FileNotFoundError(
                    f"未找到 BERT 模型文件：{bert_path}\n"
                    f"请确认模型位于 .cache/google-bert/bert-base-uncased。"
                )
        
        sam_repo_root = third_party_dir / "Grounded-SAM-2" / "sam2"

        # SAM / SAM2 默认路径
        if self.sam.model_path:
            self.sam.model_path = self._resolve_path(self.sam.model_path)
        else:
            sam2_weight = (cache_dir / "sam2.1_hiera_large.pt").resolve()
            if sam2_weight.exists():
                self.sam.model_path = str(sam2_weight)
            else:
                sam_weight = (cache_dir / "sam_vit_h_4b8939.pth").resolve()
                if sam_weight.exists():
                    self.sam.model_path = str(sam_weight)
                else:
                    raise FileNotFoundError(
                    f"未找到 SAM 权重：{sam2_weight} 或 {sam_weight}\n"
                    f"请确认权重位于 .cache 目录。"
                    )
        
        # SAM2 配置文件（Hydra 需要相对路径）
        if self.sam.config_path:
            sam_config_value = self.sam.config_path
        elif self.sam.model_type.startswith("sam2"):
            sam_config_value = "configs/sam2.1/sam2.1_hiera_l.yaml"
        else:
            sam_config_value = ""

        if sam_config_value:
            config_path = Path(sam_config_value)
            if config_path.is_absolute():
                abs_config_path = config_path.expanduser().resolve()
                try:
                    rel_config_path = abs_config_path.relative_to(sam_repo_root)
                except ValueError as exc:
                    raise ValueError(
                        f"SAM 配置文件必须位于 {sam_repo_root} 下，当前路径 {abs_config_path} 无法被 Hydra 使用。"
                    ) from exc
            else:
                rel_config_path = config_path
                abs_config_path = (sam_repo_root / rel_config_path).expanduser().resolve()

            if not abs_config_path.exists():
                raise FileNotFoundError(
                    f"未找到 SAM2 配置文件：{abs_config_path}\n"
                    f"请确认配置文件存在于 third_party/Grounded-SAM-2/sam2/configs。"
                )

            self.sam.config_path = str(rel_config_path).replace(os.sep, "/")
            self.sam.resolved_config_path = str(abs_config_path)
        else:
            self.sam.config_path = ""
            self.sam.resolved_config_path = None
        
        # Co-Tracker 默认路径
        if self.tracker.cotracker_checkpoint:
            self.tracker.cotracker_checkpoint = self._resolve_path(self.tracker.cotracker_checkpoint)
        elif self.tracker.enable_cotracker_validation:
            cotracker_weight = (cache_dir / "scaled_offline.pth").resolve()
            if cotracker_weight.exists():
                self.tracker.cotracker_checkpoint = str(cotracker_weight)
            else:
                raise FileNotFoundError(
                    f"未找到 Co-Tracker 权重：{cotracker_weight}\n"
                    f"请将权重放置于 .cache 目录，或将 tracker.enable_cotracker_validation 设为 False。"
                )
        
        # 输出目录
        self.output_dir = str(Path(self.output_dir or outputs_dir).expanduser().resolve())
        os.makedirs(self.output_dir, exist_ok=True)

        resolved_cotracker_dir = (
            Path(self.cotracker_visualization_output_dir).expanduser().resolve()
            if self.cotracker_visualization_output_dir
            else cotracker_vis_dir.expanduser().resolve()
        )
        self.cotracker_visualization_output_dir = str(resolved_cotracker_dir)
        if self.cotracker_visualization_enable:
            os.makedirs(resolved_cotracker_dir, exist_ok=True)
    
    def get_model_path(self, model_name: str) -> str:
        """
        获取指定子模块的模型路径。
        
        Args:
            model_name: 模型名称
        
        Returns:
            模型路径字符串，若未配置则返回空字符串
        """
        path_map = {
            'raft': self.raft.model_path,
            'grounding_dino': self.grounding_dino.model_path,
            'sam': self.sam.model_path,
            'tracker': self.tracker.model_path,
            'keypoint': self.keypoint.model_path
        }
        return path_map.get(model_name, "")
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """
        用传入字典更新配置。
        
        Args:
            config_dict: 配置字典
        """
        if 'raft' in config_dict:
            for key, value in config_dict['raft'].items():
                setattr(self.raft, key, value)
        
        if 'grounding_dino' in config_dict:
            for key, value in config_dict['grounding_dino'].items():
                setattr(self.grounding_dino, key, value)
        
        if 'sam' in config_dict:
            for key, value in config_dict['sam'].items():
                setattr(self.sam, key, value)
        
        if 'tracker' in config_dict:
            for key, value in config_dict['tracker'].items():
                setattr(self.tracker, key, value)
        
        if 'keypoint' in config_dict:
            for key, value in config_dict['keypoint'].items():
                setattr(self.keypoint, key, value)
        
        if 'fusion' in config_dict:
            for key, value in config_dict['fusion'].items():
                setattr(self.fusion, key, value)
        
        if 'thresholds' in config_dict:
            for key, value in config_dict['thresholds'].items():
                setattr(self.thresholds, key, value)
        
        # 更新其它配置字段
        for key, value in config_dict.items():
            if key not in ['raft', 'grounding_dino', 'sam', 'tracker', 'keypoint', 'fusion', 'thresholds']:
                if hasattr(self, key):
                    setattr(self, key, value)

        # 更新后重新标准化路径
        self.__post_init__()


def load_config_from_yaml(config_path: str) -> TemporalReasoningConfig:
    """
    从 YAML 文件加载配置。
    
    Args:
        config_path: YAML 配置文件路径
    
    Returns:
        TemporalReasoningConfig 实例
    """
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        config = TemporalReasoningConfig()
        if 'temporal_reasoning' in config_dict:
            config.update_from_dict(config_dict['temporal_reasoning'])
        else:
            config.update_from_dict(config_dict)
        
        return config
    except ImportError:
        raise ImportError("请先安装 pyyaml：pip install pyyaml")
    except Exception as e:
        raise ValueError(f"加载配置文件失败: {e}")


def get_default_config() -> TemporalReasoningConfig:
    """
    获取默认配置。
    
    Returns:
        TemporalReasoningConfig 实例
    """
    return TemporalReasoningConfig()

