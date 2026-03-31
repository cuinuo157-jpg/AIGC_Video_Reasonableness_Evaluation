# 架构说明

## 顶层流水线

```
输入视频 → FeatureHub 特征层 → 六维度分析器 → 结果聚合 → EvaluationReport
                                 ├─ D1: face_identity
                                 ├─ D2: expression_naturalness
                                 ├─ D3: biological_anomaly
                                 ├─ D4: motion_logic
                                 ├─ D5: physics_consistency
                                 └─ D6: background_consistency
```

## 数据流向

1. **输入层**: `data/` 目录存放待评测视频
2. **特征层**: `src/feature_hub/` 负责共享特征提取与缓存（光流、主体分割、人脸嵌入、关键点等）
3. **维度层**: `src/` 下按维度划分模块，每个模块消费 FeatureHub 特征并输出维度评分与细节
4. **流水线层**: `src/evaluation_pipeline.py` 统一调度六个维度并融合为总分
5. **脚本层**: `scripts/` 提供调试脚本与一键评估入口（如 `debug_bio_anomaly.py`）
6. **输出层**: `outputs/` 存放分析结果与可视化（JSON/CSV/关键帧导出）
7. **第三方**: `third_party/` 存放外部依赖模型（RAFT, SAM2, CoTracker 等）

## 模块职责（核心）

| 模块 | 职责 | 关键技术 |
|------|------|---------|
| `feature_hub` | 共享特征抽取与缓存（懒加载、本地缓存） | RAFT, InsightFace, MiDaS, SAM2, MediaPipe |
| `feature_hub.extractors.subject_segmentation` | 主体实例分割（人/前景） | Grounding DINO + SAM2, auto mask, 关键点 fallback |
| `face_identity` | 人脸身份一致性（ID 保持） | SCRFD, ArcFace, 匈牙利匹配 |
| `expression_naturalness` | 表情与 AU 自然度 | Py-Feat, 光流一致性 |
| `biological_anomaly` | 人体生物特征三级异常检测 | MediaPipe 关键点、嘴内颜色直方图、MLLM 兜底 |
| `motion_logic` | 运动逻辑与平滑度 | RAFT 光流、主体 mask（SAM2）、MLLM |
| `physics_consistency` | 物理常识与动力学一致性 | 光流、深度、MLLM 语义判定 |
| `background_consistency` | 环境/背景时序一致性 | SSIM、特征匹配、深度时序相关性 |
