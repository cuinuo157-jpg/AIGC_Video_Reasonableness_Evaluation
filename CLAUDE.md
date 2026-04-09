# AIGC Video Reasonableness Evaluation

## 项目简介

AIGC 生成式视频合理性多维度评测框架，围绕多个维度对视频进行综合打分与异常定位。集成了光流、主体分割、人脸嵌入、多模态大语言模型（DashScope）等能力，支持运动逻辑、生物特征异常、物理常识、背景一致性等多维度分析。

## 目录结构

```
├─ src/                           # 核心源码（按维度模块划分）
│  ├─ feature_hub/               # 共享特征层
│  │  ├─ extractors/             # 特征提取器（光流、深度、人脸嵌入、主体分割等）
│  │  ├─ cache.py                # 特征缓存管理
│  │  └─ hub.py                  # 特征中心协调
│  ├─ motion_logic/              # D4 运动逻辑与平滑度
│  │  ├─ dynamics_scorer.py      # 动态度评分
│  │  ├─ smoothness_scorer.py    # 平滑度评分
│  │  ├─ trajectory_curvature_scorer.py  # 轨迹曲率评分
│  │  ├─ subject_motion_scorer.py        # 主体运动评分
│  │  ├─ naturalness_judge.py    # 自然度判断（MLLM）
│  │  └─ analyzer.py             # 运动逻辑分析器
│  ├─ temporal_coherence/        # 时间一致性检测
│  │  ├─ config.py               # 配置
│  │  └─ analyzer.py             # 时间一致性分析器
│  ├─ biological_anomaly/        # D3 生物特征异常（三级检测）
│  │  ├─ eye_anomaly.py          # 眼睛异常检测
│  │  ├─ hand_anomaly.py         # 手部异常检测
│  │  ├─ mouth_anomaly.py        # 嘴部异常检测
│  │  ├─ body_anomaly.py         # 身体异常检测
│  │  ├─ mllm_bio_judge.py       # 生物特征 MLLM 判断
│  │  └─ analyzer.py             # 生物特征异常分析器
│  ├─ face_identity/             # D1 人脸身份一致性
│  │  ├─ face_tracker.py         # 人脸追踪
│  │  ├─ csim_scorer.py          # 余弦相似度评分
│  │  └─ analyzer.py             # 身份一致性分析器
│  ├─ expression_naturalness/    # D2 表情与肌肉自然度
│  │  ├─ au_extractor.py         # 动作单元提取
│  │  ├─ au_rules.py             # 动作单元规则
│  │  ├─ temporal_analysis.py    # 时间分析
│  │  └─ analyzer.py             # 表情自然度分析器
│  ├─ physics_consistency/       # D5 物理常识一致性
│  │  ├─ pixel_drift.py          # 像素漂移检测
│  │  ├─ gravity_check.py        # 重力检查
│  │  ├─ mllm_physics_judge.py   # 物理常识 MLLM 判断
│  │  └─ analyzer.py             # 物理一致性分析器
│  ├─ background_consistency/    # D6 环境/背景一致性
│  │  ├─ static_region_analysis.py  # 静态区域分析
│  │  ├─ feature_matching.py     # 特征匹配
│  │  ├─ depth_consistency.py    # 深度一致性
│  │  └─ analyzer.py             # 背景一致性分析器
│  ├─ perceptual_quality/        # 感知质量评估
│  │  ├─ blur_detection/         # 模糊检测
│  │  │  ├─ simple_blur_detector.py
│  │  │  ├─ motion_smoothness_score.py
│  │  │  └─ blur_visualization.py
│  │  └─ analyzer.py             # 感知质量分析器
│  ├─ mllm/                      # 多模态大语言模型集成
│  │  ├─ client.py               # MLLM 客户端
│  │  ├─ dashscope_video_reasonableness.py  # DashScope 视频合理性评估
│  │  ├─ config.py               # MLLM 配置
│  │  └─ prompts/                # 提示词模板
│  │     ├─ motion_naturalness.py
│  │     └─ physics_commonsense.py
│  └─ evaluation_pipeline.py     # 统一评估入口
├─ scripts/                      # 命令行脚本入口
│  ├─ debug_dynamics.py          # 运动逻辑调试
│  ├─ debug_temporal_coherence.py # 时间一致性调试
│  ├─ debug_bio_anomaly.py       # 生物特征异常调试
│  ├─ debug_expression.py        # 表情自然度调试
│  ├─ debug_face_identity.py     # 人脸身份调试
│  ├─ debug_iris_tracking.py     # 虹膜追踪调试
│  ├─ eval_video_reasonableness_dashscope.py  # DashScope 视频合理性评估
│  └─ gen_ppt_anomaly.py         # 生成异常报告 PPT
├─ tests/                        # 单元测试与集成测试
├─ data/                         # 示例/测试视频数据
├─ outputs/                      # 分析结果输出
├─ third_party/                  # 第三方模型依赖（RAFT, SAM2, CoTracker 等）
├─ docs/                         # 文档与设计说明
└─ .claude/                      # Claude Code 配置与上下文
```

## 常用命令

```bash
# 使用 uv 安装依赖（推荐）
uv sync

# 运行测试
pytest tests/ -v --tb=short

# 代码检查与格式化
ruff check src/ scripts/
ruff format src/ scripts/

# 单维度调试脚本
python scripts/debug_dynamics.py --input path/to/video.mp4 --save-vis
python scripts/debug_bio_anomaly.py --input path/to/video.mp4 --sample-rate 3 --save-vis
python scripts/debug_temporal_coherence.py --input path/to/video.mp4
python scripts/debug_expression.py --input path/to/video.mp4
python scripts/debug_face_identity.py --input path/to/video.mp4

# DashScope 视频合理性评估（需配置 API Key）
python scripts/eval_video_reasonableness_dashscope.py --input path/to/video.mp4 --output outputs/report.json

# 生成异常报告 PPT
python scripts/gen_ppt_anomaly.py --input path/to/video.mp4 --output outputs/report.pptx
```

## 开发规范

- **Python 版本**: 3.10+
- **类型注解**: 所有公开函数必须添加类型注解
- **代码风格**: 遵循 PEP 8，使用 Ruff 进行检查
- **测试**: 新增功能须附带 Pytest 测试用例，目标覆盖率 80%+
- **提交信息**: `[类型] 简要描述`，如 `[feat]`, `[fix]`, `[refactor]`, `[test]`, `[docs]`

## 关键模块说明

### Feature Hub（特征中心）
- 统一管理光流、深度、人脸嵌入、主体分割等特征提取
- 支持特征缓存，避免重复计算
- 各提取器独立配置，支持模型切换

### Motion Logic（运动逻辑）
- 基于光流与相机补偿的动态度评分
- 轨迹曲率与平滑度评分
- 集成 DashScope MLLM 进行自然度判断

### Biological Anomaly（生物特征异常）
- 三级检测：眼睛、手部、嘴部、身体
- 支持 MLLM 辅助判断
- 可视化异常区域

### MLLM 集成
- 支持 DashScope 视频 VLM
- 用于运动自然度、物理常识等高级推理
- 需配置 API Key（见 `.env.example`）

## 环境配置

创建 `.env` 文件，配置必要的 API Key：

```env
DASHSCOPE_API_KEY=your_api_key_here
```

参考 `.env.example` 了解所有可配置项。
