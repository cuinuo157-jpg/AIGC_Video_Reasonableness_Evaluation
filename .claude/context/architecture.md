# 架构说明

## 模块化架构

```
输入视频 → 预处理 → 多维度分析引擎 → 结果聚合 → 输出报告
                    ├─ temporal_reasoning   (时序合理性)
                    ├─ aux_motion_intensity (运动强度)
                    ├─ perceptual_quality   (感知质量)
                    └─ [扩展维度...]
```

## 数据流向

1. **输入层**: `data/` 目录存放待评测视频
2. **脚本层**: `scripts/` 提供命令行入口，按维度划分子目录
3. **核心层**: `src/` 包含各维度的核心算法实现
4. **输出层**: `outputs/` 存放分析结果（JSON/CSV/可视化）
5. **第三方**: `third_party/` 存放外部依赖模型（RAFT, SAM2, CoTracker 等）

## 模块职责

| 模块 | 职责 | 关键技术 |
|------|------|---------|
| `temporal_reasoning` | 时序结构一致性分析 | 光流, 实例跟踪, 关键点 |
| `aux_motion_intensity` | 运动强度与场景分类 | RAFT, CoTracker, SAM2 |
| `perceptual_quality` | 感知质量缺陷检测 | Q-Align |
