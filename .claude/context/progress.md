# 项目进度

## 已完成
- [DONE] Project structure initialized (2026-03-09)
- [DONE] 六维度模块化重构（D1–D6 对齐 `plan.md`）
- [DONE] FeatureHub 共享特征层（光流 / 人脸嵌入 / 深度 / 主体分割）
- [DONE] SAM2 + Grounding DINO 主体实例分割接入（`subject_segmentation`）
- [DONE] 生物特征异常三级检测（L1 快筛 / L2 结构 / L3 MLLM 兜底）
- [DONE] 统一评估流水线 `EvaluationPipeline` + 命令行调试脚本

## 进行中
- [ ] 工程化规范建设（CLAUDE.md, tests/, CI）
- [ ] MLLM 后端扩展与成本控制（OpenAI / Anthropic + 本地模型）

## 待规划
- [ ] 多维度评分融合策略精调（阈值、权重、严重度标定）
- [ ] 批量评测与报告生成（CSV/JSON + 可视化）
- [ ] 性能优化与加速（缓存、设备感知、批处理）
