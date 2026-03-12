# -*- coding: utf-8 -*-
"""
视频模糊检测配置模块
提供配置管理和预设配置功能
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional


class BlurDetectionConfig:
    """模糊检测配置类"""
    
    def __init__(self):
        """
        初始化配置
        
        配置包含：
        - 路径配置：基础目录、缓存目录、输出目录
        - 模型路径：各种模型的路径配置
        - 检测参数：窗口大小、阈值等
        - 可视化参数：图表样式、大小等
        - 输出参数：保存选项
        - 设备配置：计算设备、批处理大小等
        """
        # 基础路径配置
        self.base_dir = Path(__file__).parent.parent
        self.cache_dir = self.base_dir / ".cache"
        self.output_dir = self.base_dir / "视频模糊检测" / "results"
        
        # 模型路径配置
        self.model_paths: Dict[str, str] = {
            'q_align_model': str(self.cache_dir / "q-future" / "one-align"),
            'grounding_dino_config': str(
                self.base_dir / "Grounded-Segment-Anything" / "GroundingDINO" /
                "groundingdino" / "config" / "GroundingDINO_SwinB.py"
            ),
            'grounding_dino_checkpoint': str(
                self.cache_dir / "groundingdino_swinb_cogcoor.pth"
            ),
            'bert_path': str(self.cache_dir / "google-bert" / "bert-base-uncased"),
            'sam_checkpoint': str(self.cache_dir / "sam_vit_h_4b8939.pth"),
            'cotracker_checkpoint': str(self.cache_dir / "scaled_offline.pth")
        }
        
        # 检测参数配置
        self.detection_params: Dict[str, Any] = {
            'window_size': 3,  # 滑动窗口大小（帧数）
            'batch_size': 32,  # 批处理大小（用于控制内存使用）
            'blur_thresholds': {
                'mild': 0.015,      # 轻微模糊阈值
                'moderate': 0.025,   # 中等模糊阈值
                'severe': 0.04       # 严重模糊阈值
            },
            'confidence_threshold': 0.7,  # 综合置信度阈值
            'min_frames': 10,              # 最小帧数要求
            'max_frames': 1000             # 最大帧数限制
        }
        
        # 可视化配置
        self.visualization_params: Dict[str, Any] = {
            'figure_size': (15, 10),
            'dpi': 300,
            'font_size': 12,
            'color_palette': 'husl',
            'style': 'whitegrid'
        }
        
        # 输出配置
        self.output_params: Dict[str, bool] = {
            'save_visualizations': True,
            'save_detailed_reports': True,
            'save_csv_summary': True,
            'save_json_results': True
        }
        
        # 设备配置
        self.device_config: Dict[str, Any] = {
            'device': 'cuda:0',
            'batch_size': 1,
            'num_workers': 4
        }
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_name: str) -> str:
        """
        获取模型路径
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型路径，如果不存在则返回空字符串
        """
        return self.model_paths.get(model_name, "")
    
    def get_detection_param(self, param_name: str) -> Any:
        """
        获取检测参数
        
        Args:
            param_name: 参数名称
            
        Returns:
            参数值，如果不存在则返回 None
        """
        return self.detection_params.get(param_name)
    
    def get_visualization_param(self, param_name: str) -> Any:
        """
        获取可视化参数
        
        Args:
            param_name: 参数名称
            
        Returns:
            参数值，如果不存在则返回 None
        """
        return self.visualization_params.get(param_name)
    
    def get_output_param(self, param_name: str) -> Any:
        """
        获取输出参数
        
        Args:
            param_name: 参数名称
            
        Returns:
            参数值，如果不存在则返回 None
        """
        return self.output_params.get(param_name)
    
    def get_device_config(self, config_name: str) -> Any:
        """
        获取设备配置
        
        Args:
            config_name: 配置名称
            
        Returns:
            配置值，如果不存在则返回 None
        """
        return self.device_config.get(config_name)
    
    def update_model_path(self, model_name: str, new_path: str) -> None:
        """
        更新模型路径
        
        Args:
            model_name: 模型名称
            new_path: 新的模型路径
        """
        self.model_paths[model_name] = new_path
    
    def update_detection_param(self, param_name: str, new_value: Any) -> None:
        """
        更新检测参数
        
        Args:
            param_name: 参数名称
            new_value: 新的参数值
        """
        self.detection_params[param_name] = new_value
    
    def update_visualization_param(self, param_name: str, new_value: Any) -> None:
        """
        更新可视化参数
        
        Args:
            param_name: 参数名称
            new_value: 新的参数值
        """
        self.visualization_params[param_name] = new_value
    
    def update_output_param(self, param_name: str, new_value: Any) -> None:
        """
        更新输出参数
        
        Args:
            param_name: 参数名称
            new_value: 新的参数值
        """
        self.output_params[param_name] = new_value
    
    def update_device_config(self, config_name: str, new_value: Any) -> None:
        """
        更新设备配置
        
        Args:
            config_name: 配置名称
            new_value: 新的配置值
        """
        self.device_config[config_name] = new_value
    
    def validate_config(self) -> bool:
        """
        验证配置是否有效
        
        Returns:
            配置是否有效
        """
        # 检查必要的模型文件是否存在（仅检查 q_align_model，其他为可选）
        required_models = ['q_align_model']
        
        for model_name in required_models:
            model_path = self.get_model_path(model_name)
            if not model_path:
                print(f"警告: 模型路径未配置 - {model_name}")
                return False
            if not os.path.exists(model_path):
                print(f"警告: 模型文件不存在 - {model_name}: {model_path}")
                return False
        
        # 检查输出目录是否可写
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if not os.access(self.output_dir, os.W_OK):
                print(f"错误: 输出目录不可写 - {self.output_dir}")
                return False
        except Exception as e:
            print(f"错误: 无法创建输出目录 - {self.output_dir}: {e}")
            return False
        
        return True
    
    def print_config(self) -> None:
        """打印当前配置"""
        print("=== 视频模糊检测配置 ===")
        print(f"基础目录: {self.base_dir}")
        print(f"缓存目录: {self.cache_dir}")
        print(f"输出目录: {self.output_dir}")
        print("\n模型路径:")
        for name, path in self.model_paths.items():
            exists = "✓" if os.path.exists(path) else "✗"
            print(f"  {name}: {path} [{exists}]")
        print("\n检测参数:")
        for name, value in self.detection_params.items():
            print(f"  {name}: {value}")
        print("\n设备配置:")
        for name, value in self.device_config.items():
            print(f"  {name}: {value}")


# 预定义配置预设
PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    'fast': {
        'window_size': 2,
        'confidence_threshold': 0.6,
        'min_frames': 5
    },
    'accurate': {
        'window_size': 5,
        'confidence_threshold': 0.8,
        'min_frames': 20
    },
    'balanced': {
        'window_size': 3,
        'confidence_threshold': 0.7,
        'min_frames': 10
    }
}


def get_preset_config(preset_name: str = 'balanced') -> BlurDetectionConfig:
    """
    获取预定义配置
    
    Args:
        preset_name: 预设名称 ('fast', 'accurate', 'balanced')
        
    Returns:
        配置对象
    """
    config = BlurDetectionConfig()
    
    if preset_name in PRESET_CONFIGS:
        preset_params = PRESET_CONFIGS[preset_name]
        for param, value in preset_params.items():
            config.update_detection_param(param, value)
    else:
        print(f"警告: 未知的预设配置 '{preset_name}'，使用默认配置")
    
    return config


def create_custom_config(**kwargs) -> BlurDetectionConfig:
    """
    创建自定义配置
    
    Args:
        **kwargs: 配置参数
            - detection_params: 检测参数字典
            - model_paths: 模型路径字典
            - device_config: 设备配置字典
            - visualization_params: 可视化参数字典
            - output_params: 输出参数字典
            
    Returns:
        配置对象
    """
    config = BlurDetectionConfig()
    
    # 更新检测参数
    if 'detection_params' in kwargs:
        for param, value in kwargs['detection_params'].items():
            config.update_detection_param(param, value)
    
    # 更新模型路径
    if 'model_paths' in kwargs:
        for model, path in kwargs['model_paths'].items():
            config.update_model_path(model, path)
    
    # 更新设备配置
    if 'device_config' in kwargs:
        for device_param, value in kwargs['device_config'].items():
            config.update_device_config(device_param, value)
    
    # 更新可视化参数
    if 'visualization_params' in kwargs:
        for param, value in kwargs['visualization_params'].items():
            config.update_visualization_param(param, value)
    
    # 更新输出参数
    if 'output_params' in kwargs:
        for param, value in kwargs['output_params'].items():
            config.update_output_param(param, value)
    
    return config


# 默认配置实例（延迟创建）
def get_default_config() -> BlurDetectionConfig:
    """获取默认配置实例"""
    return BlurDetectionConfig()
