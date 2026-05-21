# Top5 场景 — 阈值与评分区间文档

> 针对 Top5 场景分析范围（身份一致性 / 生物特征异常 / 运动逻辑 / 物理常识 / 时间一致性）的专项阈值说明。
> 提取自全量文档 `docs/thresholds.md`。
> 生成日期: 2026-05-21

---

## 1. 总览

### 1.1 评分量纲

所有维度子分与最终综合分统一映射到 **[0, 1]** 区间，1.0 表示最优。

### 1.2 综合评分等级

| 区间 | 等级 | 英文标签 |
|------|------|----------|
| `[0.85, 1.0]` | 优秀 | excellent |
| `[0.70, 0.85)` | 良好 | good |
| `[0.50, 0.70)` | 警告 | warning |
| `[0.00, 0.50)` | 严重 | critical |

> 来源: `src/api/core.py` `_band()` 函数

### 1.3 Top5 场景维度列表

| # | 维度 Key | 权重 | 中文名 |
|---|----------|------|--------|
| 1 | `face_identity` | 0.12 | 身份一致性 |
| 2 | `biological_anomaly` | 0.15 | 生物特征异常 |
| 3 | `motion_logic` | 0.12 | 运动逻辑 |
| 4 | `physics` | 0.12 | 物理常识 |
| 5 | `temporal_coherence` | 0.10 | 时间一致性 |

> 权重来源: `src/evaluation_pipeline.py` `DEFAULT_WEIGHTS`，适用维度权重自动归一化。
> Top5 定义: `src/evaluation_pipeline.py` `DEFAULT_TOP5_TYPES`

### 1.4 维度间关系

```
┌─────────────────────────────────────────────────────────┐
│                    Top5 场景分析范围                      │
├──────────────┬──────────────┬──────────────┬────────────┤
│ 身份一致性    │ 生物特征异常   │  运动逻辑     │  物理常识   │
│ (D1)         │ (D3)         │  (D4)        │  (D5)      │
│ 人脸稳定     │ 三级检测      │  动态度+平滑  │  VLM 判定   │
├──────────────┴──────────────┴──────────────┴────────────┤
│                    时间一致性 (D7)                        │
│              目标异常出现/消失检测 (TCS-lite)              │
└─────────────────────────────────────────────────────────┘
```

D1、D3 覆盖"人"层面的异常；D4、D5 覆盖"运动/物理"层面的异常；D7 作为贯穿维度的目标一致性兜底。

---

## 2. D1 人脸身份一致性

**源码**: `src/face_identity/`

### 2.1 配置参数 (`FaceIdentityConfig`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `csim_ref_weight` | 0.4 | 参考帧余弦相似度权重 |
| `csim_adj_weight` | 0.3 | 相邻帧余弦相似度权重 |
| `csim_min_weight` | 0.2 | 最低相似度权重 |
| `drop_penalty_weight` | 0.1 | 突降惩罚权重 |
| `drop_threshold` | **0.3** | 相似度突降检测阈值 |
| `drop_window` | 3 | 突降检测窗口帧数 |

### 2.2 人脸追踪阈值

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `match_threshold` | **0.4** | 人脸嵌入余弦相似度匹配阈值 |

### 2.3 评分公式

```
identity_score = clip(
      0.4 × csim_ref      ← 与首帧的参考相似度
    + 0.3 × csim_adj      ← 相邻帧间的平均相似度
    + 0.2 × csim_min      ← 整个序列中的最低相似度
    - 0.1 × drop_penalty  ← 相似度突降惩罚
, 0, 1)
```

其中 `drop_penalty = drop_events / (embeddings_count - 1)`，突降事件定义为帧间相似度下降超过 `drop_threshold (0.3)`。

### 2.4 适用性条件

- 视频中至少检测到一张人脸
- 至少存在一条有效的人脸轨迹

---

## 3. D3 生物特征异常（三级检测）

**源码**: `src/biological_anomaly/`

### 3.1 三级权重 (`BiologicalAnomalyConfig`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `level1_weight` | 0.3 | L1 快速筛选权重 |
| `level2_weight` | 0.4 | L2 结构检测权重 |
| `level3_weight` | 0.3 | L3 MLLM 兜底权重 |

### 3.2 L1 — 眼部

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ear_blink_threshold` | **0.21** | EAR (Eye Aspect Ratio) 眨眼阈值 |
| `max_no_blink_frames` | **90** | 最大不眨眼帧数 |
| `eye_symmetry_tolerance` | **0.15** | 双眼 EAR 对称容差 |

### 3.3 L1 — 嘴部 (MAR)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mar_jump_threshold` | **0.3** | MAR (Mouth Aspect Ratio) 帧间跳变阈值 |
| `mar_sustained_open_threshold` | **0.5** | MAR 持续张开判定阈值 |
| `mar_sustained_open_max_s` | **3.0** | 最大持续张开秒数 |

### 3.4 L1 — 手部

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hand_velocity_threshold` | **0.3** | 手部关键点归一化速度阈值 |
| `hand_jitter_threshold` | **0.05** | 手部抖动幅度阈值 |
| `hand_jitter_window` | **5** | 抖动检测窗口帧数 |

### 3.5 L1 — 全身骨骼（VMBench OIS 风格）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bone_length_change_threshold` | **0.45** | 骨段长度帧间相对变化阈值（45%） |
| `angle_change_threshold` | **30.0** | 关节角度帧间变化阈值（度数） |
| `min_valid_ratio` | **0.5** | 最小有效帧比例 |

### 3.6 L2 — 结构检测

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `smoothing_window` | 3 | 滑窗平滑窗口 |
| `relative_change_threshold` | **0.45** | 相对变化率阈值 |
| `finger_count_expected` | 5 | 预期手指数量 |
| `finger_min_separation` | **0.02** | 指尖最小归一化间距（低于此值视为融合） |
| `finger_fusion_frames` | 3 | 手指融合确认连续帧数 |
| `bone_length_ratio_tolerance` | 0.15 | 骨段长度比例容差 |
| `bone_length_change_tolerance` | 0.3 | 骨段长度帧间变化容差 |
| `joint_angle_range` | (0, 180) | 关节角度允许范围 |
| `mouth_area_change_threshold` | **0.5** | 嘴内区域帧间突变阈值（比率） |
| `mouth_stability_threshold` | **0.05** | 嘴部 landmark 稳定性阈值 |
| `histogram_correlation_threshold` | **0.7** | 嘴内颜色直方图帧间相关性下限 |

### 3.7 L3 — MLLM

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_mllm` | True | 启用 MLLM 判定 |
| `mllm_max_crops` | 8 | 最大 ROI 裁剪数 |
| MLLM 异常得分 | **0.3** | 判定为异常时的分数 |
| MLLM 正常得分 | **1.0** | 判定为正常时的分数 |

### 3.8 严重程度等级 (`body_anomaly.py`)

#### 骨段长度变化严重度

| 条件 | 等级 | 实际值区间 |
|------|------|------------|
| `change > threshold × 3` | **high** (高) | > 1.35 |
| `change > threshold × 1.5` | **medium** (中) | (0.675, 1.35] |
| `change ≤ threshold × 1.5` | **low** (低) | ≤ 0.675 |

> `threshold = bone_length_change_threshold = 0.45`

#### 关节角度变化严重度

| 条件 | 等级 | 实际值区间 |
|------|------|------------|
| `diff > threshold × 3` | **high** (高) | > 90° |
| `diff > threshold × 1.5` | **medium** (中) | (45°, 90°] |
| `diff ≤ threshold × 1.5` | **low** (低) | ≤ 45° |

> `threshold = angle_change_threshold = 30.0°`

### 3.9 异常扣分规则 (`_score_from_anomalies`)

| 严重度 | 单条扣分 |
|--------|----------|
| `"low"` | **0.02** |
| `"medium"` | **0.05** |
| `"high"` | **0.10** |

### 3.10 评分公式

```
level_score = clip(1.0 - sum(per_anomaly_penalty), 0, 1)

bio_quality_score = clip(0.3 × L1 + 0.4 × L2 + 0.3 × L3, 0, 1)
```

### 3.11 检测流程

```
视频帧 → 人脸检测 → Keypoint 提取
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     L1 快速筛选    L2 结构检测    L3 MLLM 兜底
    (EAR/MAR/手速)  (手指/嘴结构)   (语义判定)
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                bio_quality_score
```

---

## 4. D4 运动逻辑与平滑度

**源码**: `src/motion_logic/`

### 4.1 配置参数 (`MotionLogicConfig`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dynamics_weight` | 0.3 | 动态度子分权重 |
| `smoothness_weight` | 0.4 | 平滑度子分权重 |
| `naturalness_weight` | 0.3 | 自然度子分权重 |
| `naturalness_smoothness_threshold` | **0.8** | MLLM 触发阈值（smoothness < 0.8 时调用） |
| `smoothness_acceleration_weight` | 0.5 | 加速度平滑权重 |
| `smoothness_trajectory_weight` | 0.5 | 轨迹曲率平滑权重 |

### 4.2 动态度 (`dynamics_scorer.py`)

基于 5+1 分量加权融合（光流幅度、空间覆盖、时序变化、空间一致性、相机因子，可选主体运动）。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `flow_threshold_dynamic` | **5.0** | 动态场景光流幅度 sigmoid 中心 |
| `flow_threshold_static` | **2.0** | 静态场景光流幅度 sigmoid 中心 |
| `flow_threshold_subject_min` | **2.0** | 有主体时的最低光流阈值 |
| `flow_subject_relief_factor` | **0.35** | 主体运动阈值放宽因子 |
| `coverage_motion_threshold` | **0.5** | 运动像素判定阈值（mag > 0.5 视为运动） |
| `temporal_std_threshold` | **0.5** | 时序标准差 sigmoid 中心 |
| `camera_score_floor` | **0.25** | 相机因子得分下限 |

### 4.3 场景类型判定

| 条件 | 类型 |
|------|------|
| `camera_magnitude > 0.5` 且 `static_ratio > 0.5` | static（静态场景） |
| 其他 | dynamic（动态场景） |

### 4.4 动态度等级

| 区间 | 等级 | 说明 |
|------|------|------|
| `[0.0, 0.2)` | 极低动态 | 纯静态画面 |
| `[0.2, 0.4)` | 低动态 | 少量微动 |
| `[0.4, 0.6)` | 中等动态 | 正常运动幅度 |
| `[0.6, 0.8)` | 高动态 | 大量运动 |
| `[0.8, 1.0]` | 极高动态 | 剧烈运动 |

### 4.5 MLLM 自然度

| 条件 | 分数 |
|------|------|
| MLLM 判定 `is_natural = True` | **1.0** |
| MLLM 判定 `is_natural = False` | **0.3** |

> MLLM 仅在 `smoothness_score < naturalness_smoothness_threshold (0.8)` 时触发，避免平滑视频的无效调用。

### 4.6 相机补偿

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `_MIN_MATCH_COUNT` | 8 | 最小 SIFT/ORB 匹配点数 |
| `_RANSAC_REPROJ_THRESHOLD` | **3.0** | RANSAC 重投影误差阈值（像素） |

### 4.7 评分公式

```
smoothness = (0.5 × flow_smoothness + 0.5 × trajectory_score) / total  (有轨迹时)
           = flow_smoothness                                          (无轨迹时)

motion_logic_score = clip(
      (0.3 × dynamics + 0.4 × smoothness + 0.3 × naturalness)    (有 MLLM)
    / (0.3 × dynamics + 0.4 × smoothness) / (0.3 + 0.4)          (无 MLLM)
, 0, 1)
```

### 4.8 适用性条件

- 至少 2 帧有效光流
- 优先使用 RAFT 光流，不可用时降级为 Farneback
- 优先使用相机补偿后的残差光流

---

## 5. D5 物理常识一致性

**源码**: `src/physics_consistency/`

### 5.1 配置参数 (`PhysicsConfig`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `drift_flow_threshold` | **0.5** | 像素漂移光流阈值 |
| `drift_min_frames` | 5 | 最小连续漂移帧数 |
| `drift_direction_tolerance` | **30.0** | 漂移方向一致性容差（度数） |
| `enable_mllm` | True | 启用 VLM 判定 |
| `drift_fallback_weight` | 1.0 | 无 VLM 时漂移评分权重 |

### 5.2 评分逻辑

```
drift_score = 1.0                                    (无漂移事件)
            = max(0.0, 1.0 - drift_events × 0.3)     (有漂移事件)

physics_score = VLM 返回的 physics_score (主路径)
              = drift_fallback_weight × drift_score  (VLM 不可用时)
```

### 5.3 双路径设计

```
                  ┌─────────────────┐
                  │  enable_mllm?   │
                  └────┬────────┬───┘
                  True │        │ False
                       ▼        ▼
              ┌───────────┐  ┌──────────────┐
              │ VLM 判定   │  │ 像素漂移降级  │
              │ (主路径)    │  │ (fast path)  │
              └───────────┘  └──────────────┘
                       │        │
                       ▼        ▼
                 physics_score (0~1)
```

VLM 判定时会将 drift_events 作为上下文注入 prompt，帮助模型聚焦于可能存在物理异常的片段。

### 5.4 适用性条件

- 至少 2 帧有效光流
- VLM 路径需 `enable_mllm=True` 且 `mllm_client` 可用

---

## 6. D7 时间一致性 (TCS-lite)

**源码**: `src/temporal_coherence/`

### 6.1 配置参数 (`TemporalCoherenceConfig`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sample_interval` | 5 | 检测帧采样间隔 |
| `min_box_area_ratio` | **0.001** | 最小检测框面积比（相对画面） |
| `iou_match_threshold` | **0.3** | 目标匹配 IoU 阈值 |
| `max_track_gap_steps` | 1 | 轨迹允许的最大间隔步数 |
| `edge_margin_ratio` | **0.08** | 画面边缘判定边距比 |
| `min_track_len_steps` | 2 | 最小有效轨迹长度 |
| `shrink_ratio_threshold` | **0.65** | 目标缩小比率阈值（< 0.65 判为缩小消失） |
| `grow_ratio_threshold` | **1.35** | 目标放大比率阈值（> 1.35 判为放大出现） |

### 6.2 事件分类逻辑

| 事件类型 | 条件 | 含义 |
|----------|------|------|
| `edge_emerge` | 边框触及画面边缘 | 目标从画面外进入（正常） |
| `edge_vanish` | 边框触及画面边缘 | 目标移出画面外（正常） |
| `small_emerge` | 面积增长 > 1.35 倍 | 目标从远处/小尺寸出现（正常） |
| `small_vanish` | 面积缩小 < 0.65 倍 | 目标远离/缩小至消失（正常） |
| `detect_gap` | 轨迹长度 < min_track_len_steps | 检测间隙（可容忍） |
| `abnormal` | 不满足以上任一条件 | **真正异常的出现/消失** |

### 6.3 评分公式

```
temporal_coherence_score = 1.0                                        (无事件)
                         = max(0.0, 1.0 - abnormal / all_events)      (有事件)
```

### 6.4 检测流程

```
视频帧 → Grounding DINO 抽样检测 → IoU 轨迹追踪 → 事件分类 → 评分
  │              │                      │              │
  │   每 sample_interval 帧采样       匹配/新建轨迹    classify 六类事件
  │   过滤 min_box_area_ratio < 0.001
```

### 6.5 适用性条件

- 至少 3 帧视频
- Grounding DINO 模型可用

---

## 附录 A. Top5 阈值速查表

| 维度 | 参数名 | 默认值 | 用途 |
|------|--------|--------|------|
| D1 | `drop_threshold` | 0.3 | 人脸相似度突降检测 |
| D1 | `match_threshold` | 0.4 | 人脸追踪匹配 |
| D3 | `ear_blink_threshold` | 0.21 | 眨眼检测 |
| D3 | `mar_jump_threshold` | 0.3 | 嘴部突变检测 |
| D3 | `mar_sustained_open_threshold` | 0.5 | 嘴部持续张开 |
| D3 | `hand_velocity_threshold` | 0.3 | 手部速度异常 |
| D3 | `hand_jitter_threshold` | 0.05 | 手部抖动异常 |
| D3 | `bone_length_change_threshold` | 0.45 | 骨骼长度变化 |
| D3 | `angle_change_threshold` | 30.0° | 关节角度变化 |
| D3 | `mouth_area_change_threshold` | 0.5 | 嘴部面积突变 |
| D3 | `histogram_correlation_threshold` | 0.7 | 嘴内颜色一致性 |
| D3 | `relative_change_threshold` | 0.45 | 通用相对变化 |
| D4 | `flow_threshold_dynamic` | 5.0 | 动态场景光流 |
| D4 | `flow_threshold_static` | 2.0 | 静态场景光流 |
| D4 | `coverage_motion_threshold` | 0.5 | 运动像素覆盖 |
| D4 | `temporal_std_threshold` | 0.5 | 时序变化 |
| D4 | `camera_score_floor` | 0.25 | 相机因子下限 |
| D4 | `naturalness_smoothness_threshold` | 0.8 | MLLM 自然度触发 |
| D5 | `drift_flow_threshold` | 0.5 | 像素漂移光流 |
| D5 | `drift_direction_tolerance` | 30.0° | 漂移方向容差 |
| D7 | `iou_match_threshold` | 0.3 | 目标匹配 IoU |
| D7 | `shrink_ratio_threshold` | 0.65 | 目标缩小消失 |
| D7 | `grow_ratio_threshold` | 1.35 | 目标放大出现 |
| D7 | `min_box_area_ratio` | 0.001 | 最小检测框面积 |
| D7 | `edge_margin_ratio` | 0.08 | 画面边缘判定边距 |

## 附录 B. MLLM 触发条件速查

| 维度 | 触发模块 | 触发条件 |
|------|----------|----------|
| D3 | `mllm_bio_judge` | L1+L2 发现疑似异常帧 |
| D4 | `naturalness_judge` | `smoothness_score < 0.8` |
| D5 | `mllm_physics_judge` | `enable_mllm=True`（始终触发） |

## 附录 C. 严重度（Severity）三级定义

| 等级 | 骨段变化条件 | 角度变化条件 |
|------|-------------|-------------|
| `"high"` | `change > 1.35` (0.45 × 3) | `diff > 90°` (30 × 3) |
| `"medium"` | `change > 0.675` (0.45 × 1.5) | `diff > 45°` (30 × 1.5) |
| `"low"` | `change ≤ 0.675` | `diff ≤ 45°` |

## 附录 D. 异常扣分速查

| 严重度 | 单条扣分 | 10 条累计 | 50 条累计 |
|--------|:------:|:------:|:------:|
| `"low"` | 0.02 | -0.20 | -1.00 (底) |
| `"medium"` | 0.05 | -0.50 | -2.50 (底) |
| `"high"` | 0.10 | -1.00 (底) | -5.00 (底) |

> 分数 clip 到 [0, 1]，超过 1.0 的总扣分全部压底。

## 附录 E. 各维度适用性前置条件

| 维度 | 前置条件 | 不满足时行为 |
|------|----------|-------------|
| D1 身份一致性 | 检测到人脸 + 存在有效轨迹 | `applicable=False`，skip |
| D3 生物特征异常 | 检测到人脸 + Keypoint 可用 | `applicable=False`，skip |
| D4 运动逻辑 | 光流 ≥ 2 帧 | `applicable=False`，skip |
| D5 物理常识 | 光流 ≥ 2 帧 | `applicable=False`，skip |
| D7 时间一致性 | 视频 ≥ 3 帧 + Grounding DINO 可用 | `applicable=False`，skip |
