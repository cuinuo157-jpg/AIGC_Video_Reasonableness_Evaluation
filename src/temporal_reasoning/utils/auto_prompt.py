"""
自动生成prompt的工具模块
支持多种无prompt评测方案
"""

from typing import List, Optional, Sequence
import numpy as np
from PIL import Image


class AutoPromptGenerator:
    """自动生成prompt的生成器"""
    
    # 通用prompt列表（覆盖常见对象类别）
    GENERIC_PROMPTS = [
        "person", "human", "face", "hand", "body",
        "animal", "dog", "cat", "bird",
        "vehicle", "car", "truck", "bicycle",
        "furniture", "chair", "table", "sofa",
        "object", "thing", "item"
    ]
    
    # 更广泛的通用prompt（用于fallback）
    FALLBACK_PROMPTS = ["object", "thing"]
    
    def __init__(
        self,
        method: str = "generic",
        ram_model_path: Optional[str] = None,
        yolo_model_path: Optional[str] = None,
    ):
        """
        初始化自动prompt生成器
        
        Args:
            method: 生成方法，可选: "generic", "ram", "yolo"
            ram_model_path: RAM模型路径（如果使用ram方法）
            yolo_model_path: YOLO模型路径（如果使用yolo方法）
        """
        self.method = method
        self.ram_model = None
        self.yolo_model = None
        
        if method == "ram" and ram_model_path:
            self._load_ram_model(ram_model_path)
        elif method == "yolo" and yolo_model_path:
            self._load_yolo_model(yolo_model_path)
    
    def _load_ram_model(self, model_path: str):
        """加载RAM模型"""
        try:
            # 这里需要根据实际的RAM模型加载方式实现
            # 示例代码，需要根据实际情况调整
            import sys
            from pathlib import Path
            
            # 假设RAM模型在third_party目录下
            # 实际使用时需要根据项目结构调整
            print(f"[AutoPrompt] 加载RAM模型: {model_path}")
            # TODO: 实现RAM模型加载
            # from ram.models import ram_model
            # self.ram_model = ram_model.load_model(model_path)
            print("[AutoPrompt] 警告: RAM模型加载未实现，回退到通用prompt")
            self.method = "generic"
        except Exception as e:
            print(f"[AutoPrompt] RAM模型加载失败: {e}，回退到通用prompt")
            self.method = "generic"
    
    def _load_yolo_model(self, model_path: str):
        """加载YOLO模型"""
        try:
            from ultralytics import YOLO
            print(f"[AutoPrompt] 加载YOLO模型: {model_path}")
            self.yolo_model = YOLO(model_path)
        except ImportError:
            print("[AutoPrompt] 警告: ultralytics未安装，无法使用YOLO方法")
            print("[AutoPrompt] 安装命令: pip install ultralytics")
            self.method = "generic"
        except Exception as e:
            print(f"[AutoPrompt] YOLO模型加载失败: {e}，回退到通用prompt")
            self.method = "generic"
    
    def generate_from_image(
        self,
        image: Image.Image,
        max_classes: int = 10,
        confidence_threshold: float = 0.3
    ) -> List[str]:
        """
        从图像自动生成prompt列表
        
        Args:
            image: PIL图像
            max_classes: 最大类别数
            confidence_threshold: 置信度阈值（仅用于YOLO）
            
        Returns:
            prompt字符串列表
        """
        if self.method == "ram":
            return self._generate_with_ram(image, max_classes)
        elif self.method == "yolo":
            return self._generate_with_yolo(image, max_classes, confidence_threshold)
        else:  # generic
            return self._generate_generic()
    
    def _generate_with_ram(self, image: Image.Image, max_classes: int) -> List[str]:
        """使用RAM模型生成prompt"""
        if self.ram_model is None:
            return self._generate_generic()
        
        try:
            # TODO: 实现RAM推理
            # tags = self.ram_model.predict(image)
            # class_list = tags.split(" | ")
            # return class_list[:max_classes]
            return self._generate_generic()
        except Exception as e:
            print(f"[AutoPrompt] RAM推理失败: {e}，回退到通用prompt")
            return self._generate_generic()
    
    def _generate_with_yolo(self, image: Image.Image, max_classes: int, confidence_threshold: float) -> List[str]:
        """使用YOLO模型生成prompt"""
        if self.yolo_model is None:
            return self._generate_generic()
        
        try:
            # 运行YOLO检测
            results = self.yolo_model(image, conf=confidence_threshold, verbose=False)
            
            # 提取检测到的类别
            detected_classes = []
            if len(results) > 0 and results[0].boxes is not None:
                # 获取类别ID
                class_ids = results[0].boxes.cls.cpu().numpy()
                # 获取类别名称
                class_names = [results[0].names[int(cid)] for cid in class_ids]
                # 去重并限制数量
                detected_classes = list(dict.fromkeys(class_names))[:max_classes]
            
            if detected_classes:
                print(f"[AutoPrompt] YOLO检测到 {len(detected_classes)} 个类别: {', '.join(detected_classes)}")
                return detected_classes
            else:
                print("[AutoPrompt] YOLO未检测到对象，使用通用prompt")
                return self._generate_generic()
        except Exception as e:
            print(f"[AutoPrompt] YOLO推理失败: {e}，回退到通用prompt")
            return self._generate_generic()
    
    def _generate_generic(self) -> List[str]:
        """生成通用prompt列表"""
        return self.GENERIC_PROMPTS.copy()
    
    def generate_fallback(self) -> List[str]:
        """生成fallback prompt（最通用的）"""
        return self.FALLBACK_PROMPTS.copy()
    
    def compose_prompt(self, prompts: Optional[Sequence[str]] = None) -> str:
        """
        组合prompt列表为单个字符串（Grounded DINO格式）
        
        Args:
            prompts: prompt列表，如果为None则使用通用prompt
            
        Returns:
            组合后的prompt字符串（以"."结尾）
        """
        if prompts is None or len(prompts) == 0:
            prompts = self._generate_generic()
        
        # 清理prompt
        cleaned_prompts = [p.strip() for p in prompts if p and p.strip()]
        
        if not cleaned_prompts:
            cleaned_prompts = self.FALLBACK_PROMPTS
        
        # 组合成字符串
        prompt = ". ".join(cleaned_prompts)
        if not prompt.endswith("."):
            prompt = f"{prompt}."
        
        return prompt


def auto_generate_prompts_from_video(
    video_path: str,
    frame_idx: int = 0,
    method: str = "generic",
    ram_model_path: Optional[str] = None,
    yolo_model_path: Optional[str] = None,
) -> List[str]:
    """
    从视频自动生成prompt（便捷函数）
    
    Args:
        video_path: 视频路径（如果为空字符串，则使用通用prompt）
        frame_idx: 用于生成prompt的帧索引（默认第一帧）
        method: 生成方法
        ram_model_path: RAM模型路径
        yolo_model_path: YOLO模型路径
        
    Returns:
        prompt列表
    """
    if not video_path or not video_path.strip():
        # 如果没有视频路径，直接使用通用prompt
        generator = AutoPromptGenerator(method="generic")
        return generator._generate_generic()
    
    try:
        from ..instance_tracking.video_io import extract_frames_from_video
        
        # 提取关键帧
        frames, _ = extract_frames_from_video(video_path)
        if not frames or frame_idx >= len(frames):
            frame_idx = 0
        
        image = Image.fromarray(frames[frame_idx])
        
        # 生成prompt
        generator = AutoPromptGenerator(
            method=method,
            ram_model_path=ram_model_path,
            yolo_model_path=yolo_model_path,
        )
        
        return generator.generate_from_image(image)
    except Exception as e:
        print(f"[AutoPrompt] 从视频生成prompt失败: {e}，使用通用prompt")
        generator = AutoPromptGenerator(method="generic")
        return generator._generate_generic()

