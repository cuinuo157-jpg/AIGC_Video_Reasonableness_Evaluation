# -*- coding: utf-8 -*-
"""
时序合理性分析执行脚本
支持参数化配置
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, List

# 添加项目根目录到路径
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.temporal_reasoning.core.temporal_analyzer import TemporalReasoningAnalyzer
from src.temporal_reasoning.core.config import (
    TemporalReasoningConfig,
    load_config_from_yaml,
    get_default_config
)
from src.temporal_reasoning.utils.video_utils import load_video_frames, get_video_info


class TemporalReasoningRunner:
    """时序合理性分析运行器"""
    
    def __init__(self, config: Optional[TemporalReasoningConfig] = None):
        """
        初始化运行器
        
        Args:
            config: 配置对象，如果为None则使用默认配置
        """
        self.config = config or get_default_config()
        self.analyzer = None
    
    def initialize(self):
        """初始化分析器"""
        self.analyzer = TemporalReasoningAnalyzer(self.config)
        self.analyzer.initialize()
    
    def run(
        self,
        video_path: str,
        text_prompts: Optional[List[str]] = None,
        output_path: Optional[str] = None
    ) -> dict:
        """
        运行分析
        
        Args:
            video_path: 视频文件路径
            text_prompts: 可选文本提示列表
            output_path: 输出结果文件路径
        
        Returns:
            分析结果字典
        """
        if self.analyzer is None:
            self.initialize()
        
        # 加载视频
        print(f"正在加载视频: {video_path}")
        video_info = get_video_info(video_path)
        print(f"视频信息: {video_info['width']}x{video_info['height']}, "
              f"{video_info['frame_count']}帧, {video_info['fps']:.2f}fps")
        
        # 加载视频帧
        video_frames = load_video_frames(video_path)
        print(f"已加载 {len(video_frames)} 帧")
        
        # 执行分析
        result = self.analyzer.analyze(
            video_frames=video_frames,
            text_prompts=text_prompts,
            fps=video_info['fps'],
            video_path=video_path  # 传入视频路径，用于可视化输出
        )
        
        # 添加视频信息
        result['video_info'] = {
            'path': video_path,
            'width': video_info['width'],
            'height': video_info['height'],
            'frame_count': video_info['frame_count'],
            'fps': video_info['fps'],
            'duration': video_info['duration']
        }
        
        # 保存结果
        if output_path:
            self._save_results(result, output_path)
        
        return result
    
    def _save_results(self, result: dict, output_path: str):
        """
        保存分析结果
        
        Args:
            result: 分析结果字典
            output_path: 输出文件路径
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 保存JSON结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {output_path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='时序合理性分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本使用（无文本提示，使用默认）
  python run_analysis.py --video path/to/video.mp4
  
  # 指定文本提示
  python run_analysis.py --video path/to/video.mp4 --prompts "person" "car"
  
  # 使用配置文件
  python run_analysis.py --video path/to/video.mp4 --config config.yaml
  
  # 指定输出路径
  python run_analysis.py --video path/to/video.mp4 --output results.json
  
  # 默认禁用关键点分析（通用物体检测模式）
  python run_analysis.py --video path/to/video.mp4
  
  # 启用关键点分析（人体分析模式，需要MediaPipe）
  python run_analysis.py --video path/to/video.mp4 --enable-keypoint
  
  # 指定设备
  python run_analysis.py --video path/to/video.mp4 --device cuda:0
        """
    )
    
    # 必需参数
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='视频文件路径'
    )
    
    # 可选参数
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径（YAML格式）'
    )
    
    parser.add_argument(
        '--prompts',
        type=str,
        nargs='+',
        default=None,
        help='文本提示列表（如: --prompts "person" "car"）。如果未指定，将使用默认提示: person, face, hand, body, object'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出结果文件路径（JSON格式）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='计算设备（如: cuda:0, cpu）'
    )
    
    parser.add_argument(
        '--raft-model',
        type=str,
        default=None,
        help='RAFT模型路径'
    )
    
    parser.add_argument(
        '--raft-type',
        type=str,
        choices=['large', 'small'],
        default=None,
        help='RAFT模型类型'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录'
    )
    
    # 结构可视化参数
    parser.add_argument(
        '--enable_structure_visualization',
        action='store_true',
        help='启用结构分支可视化（SAM2 分割 overlay）'
    )

    parser.add_argument(
        '--structure_visualization_dir',
        type=str,
        default=None,
        help='结构可视化输出目录'
    )

    parser.add_argument(
        '--structure_visualization_max_frames',
        type=int,
        default=None,
        help='结构可视化最多保存的帧数'
    )

    # CoTracker 可视化参数
    parser.add_argument(
        '--enable_cotracker_visualization',
        action='store_true',
        help='启用 CoTracker 轨迹可视化'
    )

    parser.add_argument(
        '--cotracker_visualization_dir',
        type=str,
        default=None,
        help='CoTracker 可视化输出目录'
    )

    parser.add_argument(
        '--cotracker_visualization_fps',
        type=int,
        default=None,
        help='CoTracker 可视化输出视频帧率'
    )

    parser.add_argument(
        '--cotracker_visualization_mode',
        type=str,
        choices=['rainbow', 'cool', 'optical_flow'],
        default=None,
        help='CoTracker 可视化颜色模式'
    )

    parser.add_argument(
        '--enable_cotracker',
        action='store_true',
        help='启用 CoTracker 验证（验证消失/出现的合理性，默认不启用，仅使用 SAM2 评估）'
    )

    parser.add_argument(
        '--cotracker_visualization_full_video',
        action='store_true',
        default=None,
        help='生成整体视频的追踪可视化（默认不生成）'
    )

    parser.add_argument(
        '--disable_cotracker_visualization_full_video',
        action='store_true',
        help='禁用整体视频的追踪可视化（仅生成聚焦于目标mask的可视化）'
    )

    # 光流可视化参数
    parser.add_argument(
        '--enable_motion_visualization',
        action='store_true',
        help='启用光流可视化'
    )

    parser.add_argument(
        '--motion_visualization_dir',
        type=str,
        default=None,
        help='光流可视化输出目录'
    )

    parser.add_argument(
        '--motion_visualization_max_frames',
        type=int,
        default=None,
        help='光流可视化最多保存的帧数'
    )
    
    # 关键点分析参数（默认禁用，用于通用物体检测）
    parser.add_argument(
        '--enable_keypoint',
        action='store_true',
        help='启用关键点分析（需要MediaPipe，用于人体分析）'
    )
    
    # 区域时序变化检测参数（默认开启）
    parser.add_argument(
        '--disable_region_temporal_analysis',
        action='store_true',
        help='禁用区域时序变化检测（默认开启，使用SAM2的mask进行光流和颜色相似度检测）'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 检查视频文件是否存在
    if not os.path.exists(args.video):
        print(f"错误: 视频文件不存在: {args.video}")
        sys.exit(1)
    
    # 加载配置
    if args.config:
        if not os.path.exists(args.config):
            print(f"错误: 配置文件不存在: {args.config}")
            sys.exit(1)
        try:
            config = load_config_from_yaml(args.config)
            print(f"已加载配置文件: {args.config}")
        except Exception as e:
            print(f"错误: 加载配置文件失败: {e}")
            sys.exit(1)
    else:
        config = get_default_config()
        print("使用默认配置")
    
    # 更新配置（命令行参数优先级更高）
    if args.device:
        config.device = args.device
        config.raft.use_gpu = "cuda" in args.device
    
    if args.raft_model:
        config.raft.model_path = args.raft_model
    
    if args.raft_type:
        config.raft.model_type = args.raft_type
    
    if args.output_dir:
        config.output_dir = args.output_dir

    # 更新结构检测提示词（用于 SAM2 Grounding）
    if args.prompts:
        print(f"[CLI] 传入结构检测 prompt: {', '.join(args.prompts)}")
        config.structure_prompts = args.prompts

    # 结构可视化配置
    if args.enable_structure_visualization:
        config.structure_visualization_enable = True
    if args.structure_visualization_dir:
        config.structure_visualization_output_dir = args.structure_visualization_dir
    if args.structure_visualization_max_frames is not None:
        config.structure_visualization_max_frames = max(0, args.structure_visualization_max_frames)

    # CoTracker 可视化配置
    if args.enable_cotracker_visualization:
        config.cotracker_visualization_enable = True
    if args.cotracker_visualization_dir:
        config.cotracker_visualization_output_dir = args.cotracker_visualization_dir
    if args.cotracker_visualization_fps is not None:
        config.cotracker_visualization_fps = max(1, args.cotracker_visualization_fps)
    if args.cotracker_visualization_mode:
        config.cotracker_visualization_mode = args.cotracker_visualization_mode
    
    # CoTracker 验证配置（默认不启用，仅使用SAM2评估）
    if args.enable_cotracker:
        config.enable_cotracker = True
        print("[配置] 已启用 CoTracker 验证")
    else:
        config.enable_cotracker = False
        print("[配置] CoTracker 验证已禁用（仅使用 SAM2 进行评估）")
    
    # CoTracker 整体可视化配置
    if args.disable_cotracker_visualization_full_video:
        config.cotracker_visualization_full_video = False
        print("[配置] 已禁用 CoTracker 整体视频的追踪可视化")
    elif args.cotracker_visualization_full_video is not None:
        config.cotracker_visualization_full_video = True
        print("[配置] 已启用 CoTracker 整体视频的追踪可视化")

    # 光流可视化配置
    if args.enable_motion_visualization:
        config.raft.enable_visualization = True
    if args.motion_visualization_dir:
        config.raft.visualization_output_dir = args.motion_visualization_dir
    if args.motion_visualization_max_frames is not None:
        config.raft.visualization_max_frames = max(0, args.motion_visualization_max_frames)
    
    # 关键点分析配置（默认禁用，用于通用物体检测）
    if args.enable_keypoint:
        print("[配置] 已启用关键点分析（人体分析模式，需要MediaPipe）")
        # 如果keypoint为None，则使用默认配置
        if config.keypoint is None:
            from src.temporal_reasoning.core.config import KeypointConfig
            config.keypoint = KeypointConfig()
    else:
        print("[配置] 关键点分析已禁用（通用物体检测模式）")
        config.keypoint = None  # 设置为None以禁用关键点分析
    
    # 区域时序变化检测配置（默认开启）
    if args.disable_region_temporal_analysis:
        config.enable_region_temporal_analysis = False
        print("[配置] 已禁用区域时序变化检测")
    else:
        config.enable_region_temporal_analysis = True
        print("[配置] 区域时序变化检测已启用（使用SAM2的mask进行光流和颜色相似度检测）")
    
    # 设置输出路径
    if args.output:
        output_path = args.output
    else:
        # 默认输出路径
        video_name = Path(args.video).stem
        output_dir = config.output_dir or "outputs/temporal_reasoning"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_name}_result.json")
    
    # 创建运行器
    runner = TemporalReasoningRunner(config)
    
    try:
        # 运行分析
        result = runner.run(
            video_path=args.video,
            text_prompts=args.prompts,
            output_path=output_path
        )
        
        # 打印摘要
        print("\n" + "=" * 50)
        print("分析摘要")
        print("=" * 50)
        print(f"运动合理性得分: {result['motion_reasonableness_score']:.3f}")
        print(f"结构稳定性得分: {result['structure_stability_score']:.3f}")
        print(f"异常数量: {len(result['anomalies'])}")
        
        if result['anomalies']:
            print("\n异常列表:")
            for i, anomaly in enumerate(result['anomalies'], 1):
                print(f"  {i}. [{anomaly['severity']}] {anomaly['type']} "
                      f"({anomaly['timestamp']}, 置信度: {anomaly['confidence']:.2f})")
                print(f"     描述: {anomaly['description']}")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"\n错误: 分析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

