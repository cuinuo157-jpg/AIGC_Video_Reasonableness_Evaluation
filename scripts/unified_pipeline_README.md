# 统一评测流水线使用说明

> 注意：此文档描述的是历史版 `unified_pipeline`（`temporal_reasoning` 为历史模块（已迁移））流程。  
> 当前仓库主线已迁移到 `src/evaluation_pipeline.py` + `scripts/debug_*.py`，其中时序一致性使用 `temporal_coherence` 模块。

## 概述

统一评测流水线整合了以下四个模块，实现视频的全面评测：

1. **动态度检测 (aux_motion_intensity)**: 检测视频中的运动强度
2. **模糊程度检测 (blur_new)**: 检测视频中的模糊异常
3. **时序性评估 (temporal_reasoning, 历史模块（已迁移）)**: 评估视频的时序合理性
4. **人体时序性 (region_analysis)**: 分析人体区域的时序一致性

## 特点

- **一次性视频加载**: 视频帧只加载一次，避免重复读取造成的资源浪费
- **统一输出格式**: 所有模块的结果整合到一个JSON文件中
- **批量处理支持**: 支持单个视频和批量视频处理
- **错误容错**: 单个模块失败不影响其他模块的执行

## 使用方法

### 基本用法

#### 1. 评测单个视频

```bash
python scripts/unified_pipeline.py --video path/to/video.mp4 --output results.json
```

#### 2. 批量评测视频目录

```bash
python scripts/unified_pipeline.py --video_dir path/to/videos --output_dir results/
```

#### 3. 评测多个指定视频

```bash
python scripts/unified_pipeline.py --video video1.mp4 video2.mp4 video3.mp4 --output_dir results/
```

### 高级参数

#### 设备配置

```bash
# 使用GPU
python scripts/unified_pipeline.py --video video.mp4 --device cuda:0

# 使用CPU
python scripts/unified_pipeline.py --video video.mp4 --device cpu
```

#### 模型路径配置

```bash
# 指定RAFT模型路径
python scripts/unified_pipeline.py --video video.mp4 --raft_model path/to/raft-model.pth

# 指定Q-Align模型路径
python scripts/unified_pipeline.py --video video.mp4 --q_align_model path/to/q-align-model
```

#### 处理参数

```bash
# 限制处理帧数
python scripts/unified_pipeline.py --video video.mp4 --max_frames 100

# 设置帧采样间隔（跳帧处理）
python scripts/unified_pipeline.py --video video.mp4 --frame_skip 2

# 设置批处理大小（影响模糊检测的内存使用）
python scripts/unified_pipeline.py --video video.mp4 --batch_size 16

# 设置相机视场角
python scripts/unified_pipeline.py --video video.mp4 --camera_fov 60.0
```

#### 时序性评估参数

```bash
# 指定文本提示（用于结构分析）
python scripts/unified_pipeline.py --video video.mp4 --prompts "person" "car" "hand"

# 使用自定义配置文件
python scripts/unified_pipeline.py --video video.mp4 --temporal_config config.yaml
```

#### 区域分析参数

```bash
# 只分析指定区域（mouth, left_eye, right_eye, left_hand, right_hand）
python scripts/unified_pipeline.py --video video.mp4 --regions mouth left_eye right_eye
```

### 完整示例

```bash
python scripts/unified_pipeline.py \
    --video_dir data/videos \
    --output_dir outputs/evaluation \
    --device cuda:0 \
    --batch_size 32 \
    --max_frames 200 \
    --prompts "person" "face" "hand" \
    --regions mouth left_eye right_eye
```

## 输出格式

### 单视频模式

输出JSON文件包含以下字段：

```json
{
  "video_path": "path/to/video.mp4",
  "video_name": "video",
  "timestamp": "2024-01-01 12:00:00",
  "video_info": {
    "fps": 30.0,
    "width": 1920,
    "height": 1080,
    "frame_count": 300,
    "duration": 10.0
  },
  "motion_intensity": {
    "motion_intensity": 0.75,
    "scene_type": "dynamic",
    "temporal_stats": {...},
    "component_scores": {...}
  },
  "blur_detection": {
    "blur_detected": false,
    "blur_severity": "none",
    "confidence": 0.95,
    "blur_ratio": 0.02,
    "mss_score": 0.98
  },
  "temporal_reasoning（历史模块（已迁移））": {
    "motion_reasonableness_score": 0.85,
    "structure_stability_score": 0.90,
    "anomaly_count": 2,
    "anomalies": [...]
  },
  "region_analysis": {
    "score": 0.88,
    "anomaly_count": 1,
    "anomalies": [...],
    "regions": {...}
  },
  "processing_time": 45.32,
  "status": "success"
}
```

### 批量模式

输出JSON文件包含汇总信息和所有视频的结果：

```json
{
  "total_videos": 10,
  "successful": 9,
  "failed": 1,
  "timestamp": "2024-01-01 12:00:00",
  "results": [
    {
      "video_path": "video1.mp4",
      "video_name": "video1",
      ...
    },
    ...
  ]
}
```

## 错误处理

如果某个模块执行失败，结果中会包含错误信息，但不会影响其他模块的执行：

```json
{
  "motion_intensity": {
    "error": "视频帧数不足"
  },
  "blur_detection": {
    "blur_detected": false,
    ...
  },
  ...
}
```

## 注意事项

1. **内存使用**: 处理长视频时，建议使用 `--max_frames` 限制帧数，或使用 `--frame_skip` 进行采样
2. **GPU内存**: 如果遇到GPU内存不足，可以减小 `--batch_size` 参数
3. **处理时间**: 完整评测一个视频可能需要较长时间，取决于视频长度和硬件配置
4. **模型路径**: 确保RAFT和Q-Align模型路径正确，或使用默认路径

## 依赖要求

确保已安装所有必要的依赖：

- PyTorch
- OpenCV
- NumPy
- 其他模块特定的依赖（见各模块的README）

## 常见问题

### Q: 如何处理大量视频？

A: 使用批量模式，并考虑使用 `--max_frames` 和 `--frame_skip` 来加速处理。

### Q: 某个模块一直失败怎么办？

A: 检查错误信息，确认模型路径和依赖是否正确安装。可以单独运行各模块的测试脚本进行排查。

### Q: 如何只运行部分模块？

A: 当前版本不支持选择性运行模块。如果需要，可以修改代码或使用各模块的独立脚本。

### Q: 输出文件太大怎么办？

A: 可以在代码中调整输出内容，移除不需要的详细统计信息。

## 技术细节

### 视频加载优化

流水线采用一次性加载策略：
1. 在开始处理前，一次性加载所有视频帧到内存
2. 所有模块共享这些视频帧，避免重复读取
3. 对于模糊检测模块，由于需要滑动窗口，仍需要从视频文件读取，但其他模块共享已加载的帧

### 模块执行顺序

1. 动态度检测（使用已加载的帧）
2. 模糊程度检测（从文件读取，使用滑动窗口）
3. 时序性评估（使用已加载的帧）
4. 人体时序性（使用已加载的帧）

## 更新日志

- v1.0: 初始版本，整合四个模块

