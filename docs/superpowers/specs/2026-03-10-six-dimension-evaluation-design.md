# AIGC 视频合理性六维度评测框架 — 详细设计文档

> 日期: 2026-03-10
> 状态: 已确认

---

## 1. 概述

### 1.1 目标

基于 plan.md 提出的 6 个检测维度，结合项目现有能力，设计一套完整的 AIGC 视频合理性多维度评测方案。

### 1.2 六个维度

| # | 维度 | 现有状态 | 本次工作 |
|---|------|---------|---------|
| D1 | 人脸身份一致性 | 未实现 | 全新模块 |
| D2 | 表情与肌肉运动自然度 | 部分相关 | 全新模块 |
| D3 | 生物特征细节异常 | 部分已有 | 整合 + 增强 |
| D4 | 运动逻辑与平滑度 | 大部分已有 | 整合 + MLLM 补充 |
| D5 | 物理常识与动力学一致性 | 未实现 | 全新模块 |
| D6 | 环境一致性（背景保持） | 部分已有 | 增强 + 深度图 |

### 1.3 关键设计决策

- **架构模式**: 共享基础特征层（FeatureHub）+ 维度分析层
- **MLLM 策略**: 混合策略（本地开源模型为主，商用 API 作为高精度 fallback）
- **环境兼容**: 服务器 GPU / 消费级 GPU 双环境支持，模型可配置降级
- **不适用维度处理**: 自动跳过 + 权重重分配

---

## 2. 共享基础特征层（FeatureHub）

### 2.1 设计目的

6 个维度中至少 4 个需要光流、3 个需要分割/深度信息。共享基础特征可节省 40-60% 推理时间。

### 2.2 模块结构

```
src/feature_hub/
├── hub.py              # 核心调度器，懒加载 + 缓存
├── extractors/
│   ├── optical_flow.py    # RAFT 光流（历史模块（已迁移）路径说明）
│   ├── segmentation.py    # SAM2 实例分割（复用 Grounded-SAM-2）
│   ├── depth.py           # MiDaS/ZoeDepth 深度图（新增）
│   ├── face_embedding.py  # InsightFace 人脸特征（新增）
│   ├── keypoint.py        # MediaPipe 关键点（复用 keypoint_analysis）
│   └── tracking.py        # CoTracker 点追踪（复用 co-tracker）
└── cache.py            # 特征缓存管理（内存 + 可选磁盘）
```

### 2.3 工作机制

- **懒加载**: 模型只在首次请求对应特征时加载
- **设备感知**: 自动检测 GPU 显存，显存不足时卸载已完成模型再加载新模型
- **向后兼容**: 现有模块不需改动，FeatureHub 在 unified_pipeline 层面做协调
- **可选磁盘缓存**: 批量处理时可将特征序列化到磁盘，避免重复计算

### 2.4 调用方式

```python
hub = FeatureHub(video_path, device="cuda")
flow = hub.get("optical_flow")       # 首次调用触发推理
flow = hub.get("optical_flow")       # 缓存命中
depth = hub.get("depth")             # 按需触发新模型
```

---

## 3. 维度 1: 人脸身份一致性（Face ID Consistency）

### 3.1 核心指标: CSIM

采用业界标准的 CSIM（Cosine Similarity Identity Metric），与 Face Consistency Benchmark、ConsisID (CVPR 2025)、WB-DH Benchmark 等评测基准对齐。

### 3.2 算法流程

```
视频帧序列
  → [1] 人脸检测与对齐（InsightFace 内置 SCRFD）
  → [2] ArcFace embedding 提取（512-d normalized）
  → [3] 跨帧身份关联（匈牙利算法 + 余弦相似度匹配）
  → [4] CSIM 多模式评分
```

### 3.3 CSIM 四模式

| 模式 | 计算方法 | 检测目标 |
|------|---------|---------|
| CSIM-Ref | 所有帧 vs 首帧余弦相似度均值 | 全局身份保持 |
| CSIM-Adj | 相邻帧对余弦相似度均值 | 时序平滑度 |
| CSIM-Min | 全轨迹最小余弦相似度 | 最差情况 |
| CSIM-Drop | 滑动窗口检测相似度骤降 | 突变帧定位 |

### 3.4 综合评分

```
identity_score = 0.4 * csim_ref + 0.3 * csim_adj + 0.2 * csim_min - 0.1 * drop_penalty
```

### 3.5 可选双骨干

- ArcFace（默认）: 对光照/角度变化更鲁棒
- CurricularFace: 对表情变化更敏感

均通过 InsightFace model_zoo 加载预训练权重，无需训练。

### 3.6 模块结构

```
src/face_identity/
├── __init__.py
├── analyzer.py            # FaceIdentityAnalyzer 主入口
├── face_tracker.py        # 检测 + 对齐 + 跨帧匹配
├── csim_scorer.py         # CSIM 多模式评分
└── config.py
```

### 3.7 环境兼容

| 环境 | 模型 | 显存 |
|------|------|------|
| 服务器 GPU | ArcFace-R100 + SCRFD-10G | ~2GB |
| 消费级 GPU | ArcFace-R50 + SCRFD-2.5G | ~800MB |
| CPU fallback | ArcFace-R18 + SCRFD-500M | 无 GPU |

---

## 4. 维度 2: 表情与肌肉运动自然度（Facial Expression Naturalness）

### 4.1 工具选型: Py-Feat

选择 Py-Feat 而非 OpenFace，理由:
- pip 安装，跨平台（Win/Linux/Mac）
- 20+ AU 强度回归，预训练模型内置
- PyTorch backend 支持 GPU 加速

### 4.2 算法流程（三条子路径）

**[2a] AU 组合合理性**
- 基于 FACS 规则库判定 AU 组合是否符合自然表情
- 关键规则: 真笑=AU6+AU12，假笑=AU12 无 AU6，恐怖谷效应检测

**[2b] AU 时序平滑度**
- 对每个 AU 强度时间序列计算平滑度
- 自然表情: 渐进变化；AI 伪造: 突变或锯齿波形

**[2c] 光流辅助验证**
- 面部区域光流 magnitude 与 AU 强度变化是否一致
- 表情变化大但光流小 → 纹理闪烁而非真实运动

### 4.3 AU 组合规则库

```python
NATURAL_EXPRESSIONS = {
    "genuine_smile":  {"required": ["AU6", "AU12"], "forbidden": []},
    "surprise":       {"required": ["AU1", "AU2", "AU5", "AU26"], "forbidden": []},
    "frown":          {"required": ["AU4"], "forbidden": ["AU12"]},
    "fear":           {"required": ["AU1", "AU2", "AU4", "AU20"], "forbidden": []},
}
CONFLICT_PAIRS = [("AU1+AU2", "AU4"), ("AU23", "AU26")]
```

### 4.4 模块结构

```
src/expression_naturalness/
├── __init__.py
├── analyzer.py          # ExpressionAnalyzer 主入口
├── au_extractor.py      # Py-Feat AU 提取封装
├── au_rules.py          # FACS 规则库 + 组合合理性判定
├── temporal_analysis.py # AU 时序平滑度分析
└── config.py
```

---

## 5. 维度 3: 生物特征细节异常（Biological Feature Anomaly）

### 5.1 设计思路

复用现有 keypoint_analysis 能力，在其上添加规则引擎层。

### 5.2 三个检测项

**[3a] 眼睛异常**
- EAR 异常: 长期不眨眼（>90帧）或双眼不同步
- 瞳孔追踪: MediaPipe Iris → 瞳孔固定不动 / 左右眼注视方向不一致
- 对称性: 双眼开合度差异超阈值

**[3b] 手指畸变**
- 手指数量 ≠ 5
- 关节角度超出人体工学范围 (0°~180°)
- 相邻帧手指骨骼长度比突变（容忍度 15%）
- 手指骨骼线段自穿透

**[3c] 口腔异常**
- 口腔区域分割 → 牙齿区域像素分析
- 牙齿数量/排列异常（边缘检测 + 连通域）

### 5.3 人体工学约束规则库

```python
HAND_CONSTRAINTS = {
    "finger_count": 5,
    "joint_angle_range": (0, 180),
    "thumb_angle_range": (-30, 130),
    "bone_length_ratio_tolerance": 0.15,
}
EYE_CONSTRAINTS = {
    "ear_blink_threshold": 0.21,
    "max_no_blink_frames": 90,
    "symmetry_tolerance": 0.15,
}
```

### 5.4 模块结构

```
src/biological_anomaly/
├── __init__.py
├── analyzer.py              # BiologicalAnomalyAnalyzer 主入口
├── eye_anomaly.py           # 眼睛异常检测
├── hand_anomaly.py          # 手指畸变检测
├── mouth_anomaly.py         # 口腔异常检测
├── anomaly_rules.py         # 人体工学约束规则库
└── config.py
```

---

## 6. 维度 4: 运动逻辑与平滑度（Motion Logic & Smoothness）

### 6.1 整合策略

整合三个已有模块 + MLLM 补充:
- aux_motion_intensity → 动态度评分
- aux_motion_intensity_2 → PAS 评分
- perceptual_quality → 运动平滑度
- MLLM → 运动自然度（新增）

### 6.2 运动平滑度增强

现有 Q-Align MSS 效果较差，增强为三信号融合:

| 信号 | 来源 | 检测目标 |
|------|------|---------|
| Q-Align MSS | 保留作为基线 | 全局质量感知 |
| 光流加速度 | FeatureHub optical_flow 二阶导数 | 瞬间跳变 |
| 轨迹曲率 | FeatureHub tracking 曲率变化率 | 物体瞬移 |

### 6.3 运动自然度（MLLM 辅助）

```
预筛选（光流/平滑度异常片段）
  → 候选片段采样（8~16 帧）
  → MLLM 推理（structured JSON 输出）
  → 整合到 motion_naturalness_score
```

本地模型: InternVL2-8B / Qwen2-VL-7B
API fallback: GPT-4o / Claude

### 6.4 MLLM 统一调用层

```
src/mllm/
├── client.py       # MLLMClient: 统一接口，支持 local/api/hybrid
└── prompts/        # 各维度的 prompt 模板
```

供维度 4（运动自然度）和维度 5（物理常识）共用。

### 6.5 模块结构

```
src/motion_logic/
├── __init__.py
├── analyzer.py               # MotionLogicAnalyzer 主入口
├── dynamics_scorer.py        # 动态度评分（封装 PAS + motion_intensity）
├── smoothness_scorer.py      # 平滑度增强版（三信号融合）
├── naturalness_judge.py      # 运动自然度（MLLM 辅助）
└── config.py
```

---

## 7. 维度 5: 物理常识与动力学一致性（Physics Commonsense）

### 7.1 设计策略

算法可检测的先做 + MLLM 兜底语义判定。确定性由高到低:

### 7.2 像素漂移检测（算法，可靠）

```
静态区域识别（光流阈值 or SAM 背景分割）
  → 静态区域内微运动检测
  → 轨迹闭合度分析（闭合=微抖动，非闭合+单向=漂移）
  → drift_events[], drift_severity
```

### 7.3 重力方向一致性（算法，中等）

```
追踪自由运动物体轨迹（CoTracker）
  → y 分量二次拟合（抛物线检验）
  → 判定是否符合重力加速度方向
  → 需结合深度图判断相机俯仰角
```

### 7.4 MLLM 物理常识判定（语义，实验性）

覆盖算法无法检测的场景: 水往高处流、刚体穿模、影子方向不一致、物体凭空出现/消失。

### 7.5 实现优先级

像素漂移（可靠）> 重力检测（中等）> MLLM 物理判定（实验性）

### 7.6 模块结构

```
src/physics_consistency/
├── __init__.py
├── analyzer.py              # PhysicsConsistencyAnalyzer 主入口
├── pixel_drift.py           # 像素漂移检测
├── gravity_check.py         # 重力方向一致性
├── mllm_physics_judge.py    # MLLM 物理常识判定
└── config.py
```

---

## 8. 维度 6: 环境一致性（Background Consistency）

### 8.1 三层检测架构

**[6a] 静态区域残差分析**
- 前景/背景分离（光流阈值 or SAM 分割）
- 背景区域: 像素残差 + 颜色直方图一致性 + SSIM 局部对比
- 检测: 背景突变、色调漂移、结构退化

**[6b] 特征点匹配 + 单应性**
- 背景 SIFT/ORB 特征点匹配 → 单应性矩阵 H
- H 突变或匹配点过少 → 背景不一致
- 补偿相机运动后的残差分析

**[6c] 深度图时序一致性（新增）**
- MiDaS/ZoeDepth 逐帧深度图
- 相邻帧深度图相关系数（闪烁检测）
- 深度边缘与 RGB 边缘对齐度
- 空间结构连通性（深度断层检测）

### 8.2 深度模型选型

| 模型 | 精度 | 速度 | 显存 | 场景 |
|------|------|------|------|------|
| MiDaS DPT-Hybrid | 中 | 快 | ~1GB | 消费级 GPU 默认 |
| MiDaS DPT-Large | 高 | 中 | ~2GB | 服务器默认 |
| Depth Anything v2 | 高 | 快 | ~1.5GB | 综合最优备选 |

### 8.3 三层信号融合

```
background_score = 0.3 * residual_score       # 像素级
               + 0.3 * homography_stability   # 几何级
               + 0.4 * depth_consistency      # 结构级
```

### 8.4 模块结构

```
src/background_consistency/
├── __init__.py
├── analyzer.py                # BackgroundConsistencyAnalyzer 主入口
├── static_region_analysis.py  # 静态区域掩码 + 残差分析
├── feature_matching.py        # 特征点匹配 + 单应性
├── depth_consistency.py       # 深度图时序一致性
└── config.py
```

---

## 9. 统一流水线集成

### 9.1 架构

```
输入视频
  → FeatureHub 初始化
  → 六维度分析（可配置启用/禁用）
  → 不适用维度自动跳过 + 权重重分配
  → 综合评分 + 结构化报告
```

### 9.2 维度适用性检测

| 维度 | 适用条件 | 跳过条件 |
|------|---------|---------|
| D1 人脸身份 | 检测到人脸 | 无人脸 |
| D2 表情自然度 | 检测到人脸 | 无人脸 |
| D3 生物特征 | 检测到人脸或手 | 无人脸且无手 |
| D4 运动逻辑 | 存在运动 | 全静态 |
| D5 物理常识 | 存在运动物体 | 无运动物体 |
| D6 环境一致性 | 始终适用 | 不跳过 |

### 9.3 权重重分配

不适用维度标记为 N/A，剩余维度权重按原始比例归一化。

### 9.4 配置化

```yaml
evaluation:
  dimensions:
    face_identity:      { enabled: true,  weight: 0.20 }
    expression:         { enabled: true,  weight: 0.15 }
    biological_anomaly: { enabled: true,  weight: 0.10 }
    motion_logic:       { enabled: true,  weight: 0.25 }
    physics:            { enabled: true,  weight: 0.15 }
    background:         { enabled: true,  weight: 0.15 }
  feature_hub:
    device: "auto"
    cache_to_disk: false
    max_gpu_memory_gb: 8
  mllm:
    backend: "local"
    local_model: "InternVL2-8B"
    api_provider: "openai"
    api_model: "gpt-4o"
```

### 9.5 输出

```python
@dataclass
class DimensionResult:
    applicable: bool
    skip_reason: str | None
    score: float | None
    weight: float
    details: Any

@dataclass
class EvaluationReport:
    dimensions: dict[str, DimensionResult]
    active_dimensions: list[str]
    final_score: float
    video_type: str   # "human", "scene", "object", "mixed"
```

---

## 10. 完整模块结构总览

```
src/
├── feature_hub/                  # [新增] 共享基础特征层
│   ├── hub.py
│   ├── extractors/
│   └── cache.py
├── face_identity/                # [新增] D1: 人脸身份一致性
├── expression_naturalness/       # [新增] D2: 表情自然度
├── biological_anomaly/           # [新增] D3: 生物特征异常
├── motion_logic/                 # [新增] D4: 运动逻辑（整合已有）
├── physics_consistency/          # [新增] D5: 物理常识
├── background_consistency/       # [新增] D6: 环境一致性（增强已有）
├── mllm/                         # [新增] MLLM 统一调用层
├── temporal_reasoning/           # 历史模块（已迁移）
├── aux_motion_intensity/         # [保留] 被 motion_logic 封装
├── aux_motion_intensity_2/       # [保留] 被 motion_logic 封装
└── perceptual_quality/           # [保留] 被 motion_logic 封装
```
