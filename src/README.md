# src — 源码根目录

AIGC 视频合理性评测框架的核心源码，按评测维度拆分为独立子包。

## 目录概览

| 目录 | 说明 |
|------|------|
| `feature_hub/` | 共享特征层（光流/深度/人脸/分割/追踪提取器 + 缓存） |
| `face_identity/` | D1 人脸身份一致性 |
| `expression_naturalness/` | D2 表情与肌肉自然度 |
| `biological_anomaly/` | D3 生物特征异常（三级检测） |
| `motion_logic/` | D4 运动逻辑与平滑度 |
| `physics_consistency/` | D5 物理常识一致性 |
| `background_consistency/` | D6 环境/背景一致性 |
| `temporal_coherence/` | D7 时间一致性（TCS-lite） |
| `perceptual_quality/` | 感知质量评估（模糊/瑕疵检测） |
| `mllm/` | 多模态大语言模型集成（DashScope/VLLM） |
| `api/` | FastAPI 评测服务 + 任务管理 |
| `webui/` | Web 可视化界面服务 |
| `evaluation_pipeline.py` | 统一评测入口（8 维度 + 3 种分析范围） |

## 开发规范

- 子包之间不应直接互相导入，公共工具放在 `src/` 根级别
- 公开函数必须添加类型注解和 docstring
- 内部辅助函数以 `_` 开头
- 遵循 PEP 8，使用 Ruff 检查
