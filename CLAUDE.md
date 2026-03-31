# AIGC Video Reasonableness Evaluation

## 项目简介

AIGC 生成式视频合理性多维度评测框架，围绕六个维度（身份一致性、表情自然度、生物特征异常、运动逻辑、物理常识、环境一致性）对视频进行综合打分与异常定位。

## 目录结构

```
├─ src/                       # 核心源码（按维度模块划分）
│  ├─ feature_hub/           # 共享特征层（光流 / 主体分割 / 人脸嵌入 / 深度）
│  ├─ face_identity/         # D1 人脸身份一致性
│  ├─ expression_naturalness/ # D2 表情与肌肉自然度
│  ├─ biological_anomaly/    # D3 生物特征异常（三级检测）
│  ├─ motion_logic/          # D4 运动逻辑与平滑度
│  ├─ physics_consistency/   # D5 物理常识一致性
│  ├─ background_consistency/# D6 环境/背景一致性
│  └─ evaluation_pipeline.py # 统一评估入口
├─ scripts/                   # 命令行脚本入口（单维度调试 / 统一评估）
├─ tests/                     # 单元测试与集成测试
├─ data/                      # 示例/测试视频数据
├─ outputs/                   # 分析结果输出
├─ third_party/               # 第三方模型依赖（RAFT, SAM2, CoTracker 等）
└─ .claude/                   # Claude Code 配置与上下文
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

# 单维度调试（示例：生物特征异常）
python scripts/debug_bio_anomaly.py --input path/to/video.mp4 --sample-rate 3 --save-vis

# 运行统一流水线（如果已提供）
python scripts/evaluate.py --input path/to/video.mp4 --output outputs/report.json
```

## 开发规范

- **Python 版本**: 3.10+
- **类型注解**: 所有公开函数必须添加类型注解
- **代码风格**: 遵循 PEP 8，使用 Ruff 进行检查
- **测试**: 新增功能须附带 Pytest 测试用例
- **提交信息**: `[类型] 简要描述`，如 `[add]`, `[fix]`, `[refactor]`
