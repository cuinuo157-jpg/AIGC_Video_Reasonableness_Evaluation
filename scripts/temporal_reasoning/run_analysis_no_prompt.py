#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无Prompt评测示例脚本

演示如何使用自动prompt生成功能进行视频评测，无需手动提供prompt。
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.temporal_reasoning.core.config import TemporalReasoningConfig
from src.temporal_reasoning.core.temporal_analyzer import TemporalAnalyzer
from src.temporal_reasoning.core.video_utils import load_video_frames, get_video_info


def main():
    parser = argparse.ArgumentParser(
        description="无Prompt视频评测示例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用通用prompt（默认方法）
  python run_analysis_no_prompt.py --video path/to/video.mp4
  
  # 使用YOLO自动检测对象
  python run_analysis_no_prompt.py --video path/to/video.mp4 --auto-method yolo
  
  # 使用自定义通用prompt列表
  python run_analysis_no_prompt.py --video path/to/video.mp4 \\
      --generic-prompts "person" "car" "dog"
        """
    )
    parser.add_argument("--video", required=True, help="视频文件路径")
    parser.add_argument(
        "--auto-method",
        choices=["generic", "yolo", "ram"],
        default="generic",
        help="自动prompt生成方法（默认: generic）"
    )
    parser.add_argument(
        "--yolo-model",
        default="yolov8n.pt",
        help="YOLO模型路径（如果使用yolo方法，默认: yolov8n.pt）"
    )
    parser.add_argument(
        "--ram-model",
        help="RAM模型路径（如果使用ram方法）"
    )
    parser.add_argument(
        "--generic-prompts",
        nargs="+",
        help="自定义通用prompt列表（如果使用generic方法）"
    )
    parser.add_argument(
        "--output",
        help="输出结果JSON文件路径（可选）"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="计算设备（默认: cuda:0）"
    )
    
    args = parser.parse_args()
    
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 创建配置
    config = TemporalReasoningConfig()
    config.device = args.device
    
    # 启用自动prompt生成
    config.enable_auto_prompt = True
    config.auto_prompt_method = args.auto_method
    
    if args.auto_method == "yolo":
        config.auto_prompt_yolo_model_path = args.yolo_model
        print(f"[配置] 使用YOLO自动检测，模型: {args.yolo_model}")
    elif args.auto_method == "ram":
        if args.ram_model:
            config.auto_prompt_ram_model_path = args.ram_model
            print(f"[配置] 使用RAM自动检测，模型: {args.ram_model}")
        else:
            print("[警告] RAM方法需要指定模型路径，回退到generic方法")
            config.auto_prompt_method = "generic"
    
    if args.generic_prompts:
        config.generic_prompts = args.generic_prompts
        print(f"[配置] 使用自定义通用prompt: {', '.join(args.generic_prompts)}")
    
    # 创建分析器
    print("\n" + "=" * 60)
    print("初始化时序合理性分析器（无Prompt模式）")
    print("=" * 60)
    analyzer = TemporalAnalyzer(config)
    
    # 加载视频
    print(f"\n正在加载视频: {video_path.name}")
    video_info = get_video_info(str(video_path))
    print(f"视频信息: {video_info['width']}x{video_info['height']}, "
          f"{video_info['frame_count']}帧, {video_info['fps']:.2f}fps")
    
    video_frames = load_video_frames(str(video_path))
    print(f"已加载 {len(video_frames)} 帧")
    
    # 执行分析（不提供text_prompts参数，让系统自动生成）
    print(f"\n开始分析视频: {video_path.name}")
    print(f"自动prompt方法: {config.auto_prompt_method}")
    print("-" * 60)
    
    result = analyzer.analyze(
        video_frames=video_frames,
        text_prompts=None,  # 不提供prompt，触发自动生成
        fps=video_info['fps'],
        video_path=str(video_path),
    )
    
    # 输出结果
    print("\n" + "=" * 60)
    print("分析完成")
    print("=" * 60)
    print(f"运动合理性分数: {result.get('motion_reasonableness_score', 0.0):.4f}")
    print(f"结构稳定性分数: {result.get('structure_stability_score', 0.0):.4f}")
    print(f"异常数量: {len(result.get('anomalies', []))}")
    
    # 保存结果
    if args.output:
        import json
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n结果已保存到: {output_path}")
    else:
        # 默认保存到outputs目录
        output_dir = project_root / "outputs" / "temporal_reasoning"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_path.stem}_no_prompt_result.json"
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()

