# AIGC Video Reasonableness Evaluation — 阈值与评分区间文档

> 自动提取自源码，覆盖所有评测维度的阈值参数、评分范围与等级划分。
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

### 1.3 维度默认权重

| 维度 | 权重 | 维度 Key |
|------|------|----------|
| 人脸身份一致性 (D1) | 0.12 | `face_identity` |
| 表情自然度 (D2) | 0.12 | `expression` |
| 生物特征异常 (D3) | 0.15 | `biological_anomaly` |
| 运动逻辑与平滑度 (D4) | 0.12 | `motion_logic` |
| 时间一致性 (D7) | 0.10 | `temporal_coherence` |
| 物理常识一致性 (D5) | 0.12 | `physics` |
| 背景一致性 (D6) | 0.15 | `background` |
| 感知质量 | 0.12 | `perceptual_quality` |

> 来源: `src/evaluation_pipeline.py` `DEFAULT_WEIGHTS`，适用维度权重自动归一化。

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
      0.4 × csim_ref
    + 0.3 × csim_adj
    + 0.2 × csim_min
    - 0.1 × drop_penalty
, 0, 1)
```

其中 `drop_penalty = drop_events / (embeddings_count - 1)`，突降事件定义为帧间相似度下降超过 `drop_threshold (0.3)`。

---

## 3. D2 表情与肌肉自然度

**源码**: `src/expression_naturalness/`

### 3.1 配置参数 (`ExpressionConfig`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `au_smoothness_window` | 5 | AU 平滑窗口帧数 |
| `au_jump_threshold` | **1.5** | AU 值跳变阈值 |
| `flow_consistency_weight` | 0.3 | 光流一致性权重 |
| `au_combination_weight` | 0.4 | AU 组合规则权重 |
| `au_smoothness_weight` | 0.3 | AU 时序平滑度权重 |

### 3.2 AU 规则阈值 (`au_rules.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `AU_ACTIVATION_THRESHOLD` | **1.0** | AU 激活判定阈值 |

### 3.3 AU 平滑度归一化 (`temporal_analysis.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_possible_diff` | **5.0** | AU 逐帧最大可能差值（归一化分母） |

### 3.4 评分公式

```
smoothness = clip(1.0 - mean(|AU_diff| / 5.0), 0, 1)
violation_penalty = min(violations / frames, 1.0)
expression_score = clip(
      0.4 × (1.0 - violation_penalty)
    + 0.3 × temporal_smoothness
    + 0.3 × 1.0  (flow_consistency, 占位)
, 0, 1)
```

---

## 4. D3 生物特征异常（三级检测）

**源码**: `src/biological_anomaly/`

### 4.1 三级权重 (`BiologicalAnomalyConfig`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `level1_weight` | 0.3 | L1 快速筛选权重 |
| `level2_weight` | 0.4 | L2 结构检测权重 |
| `level3_weight` | 0.3 | L3 MLLM 兜底权重 |

### 4.2 L1 — 眼部

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ear_blink_threshold` | **0.21** | EAR (Eye Aspect Ratio) 眨眼阈值 |
| `max_no_blink_frames` | **90** | 最大不眨眼帧数 |
| `eye_symmetry_tolerance` | **0.15** | 双眼 EAR 对称容差 |

### 4.3 L1 — 嘴部 (MAR)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mar_jump_threshold` | **0.3** | MAR (Mouth Aspect Ratio) 帧间跳变阈值 |
| `mar_sustained_open_threshold` | **0.5** | MAR 持续张开判定阈值 |
| `mar_sustained_open_max_s` | **3.0** | 最大持续张开秒数 |

### 4.4 L1 — 手部

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hand_velocity_threshold` | **0.3** | 手部关键点归一化速度阈值 |
| `hand_jitter_threshold` | **0.05** | 手部抖动幅度阈值 |
| `hand_jitter_window` | **5** | 抖动检测窗口帧数 |

### 4.5 L1 — 全身骨骼（VMBench OIS 风格）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bone_length_change_threshold` | **0.45** | 骨段长度帧间相对变化阈值（45%） |
| `angle_change_threshold` | **30.0** | 关节角度帧间变化阈值（度数） |
| `min_valid_ratio` | **0.5** | 最小有效帧比例 |

### 4.6 L2 — 结构检测

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

### 4.7 L3 — MLLM

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_mllm` | True | 启用 MLLM 判定 |
| `mllm_max_crops` | 8 | 最大 ROI 裁剪数 |
| MLLM 异常得分 | **0.3** | 判定为异常时的分数 |
| MLLM 正常得分 | **1.0** | 判定为正常时的分数 |

### 4.8 严重程度等级 (`body_anomaly.py`)

#### 骨段长度变化严重度

| 条件 | 等级 |
|------|------|
| `change > threshold × 3` | **high** (高) |
| `change > threshold × 1.5` | **medium** (中) |
| `change ≤ threshold × 1.5` | **low** (低) |

> 其中 `threshold = bone_length_change_threshold = 0.45`，即 high > 1.35，medium > 0.675

#### 关节角度变化严重度

| 条件 | 等级 |
|------|------|
| `diff > threshold × 3` | **high** (高) |
| `diff > threshold × 1.5` | **medium** (中) |
| `diff ≤ threshold × 1.5` | **low** (低) |

> 其中 `threshold = angle_change_threshold = 30.0°`，即 high > 90°，medium > 45°

### 4.9 异常扣分规则 (`_score_from_anomalies`)

| 严重度 | 单条扣分 |
|--------|----------|
| `"low"` | **0.02** |
| `"medium"` | **0.05** |
| `"high"` | **0.10** |

```
level_score = clip(1.0 - sum(per_anomaly_penalty), 0, 1)
bio_quality_score = clip(0.3 × L1 + 0.4 × L2 + 0.3 × L3, 0, 1)
```

---

## 5. D4 运动逻辑与平滑度

**源码**: `src/motion_logic/`

### 5.1 配置参数 (`MotionLogicConfig`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dynamics_weight` | 0.3 | 动态度子分权重 |
| `smoothness_weight` | 0.4 | 平滑度子分权重 |
| `naturalness_weight` | 0.3 | 自然度子分权重 |
| `naturalness_smoothness_threshold` | **0.8** | MLLM 触发阈值（smoothness < 0.8 时调用） |
| `smoothness_acceleration_weight` | 0.5 | 加速度平滑权重 |
| `smoothness_trajectory_weight` | 0.5 | 轨迹曲率平滑权重 |

### 5.2 动态度 (`dynamics_scorer.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `flow_threshold_dynamic` | **5.0** | 动态场景光流幅度 sigmoid 中心 |
| `flow_threshold_static` | **2.0** | 静态场景光流幅度 sigmoid 中心 |
| `flow_threshold_subject_min` | **2.0** | 有主体时的最低光流阈值 |
| `flow_subject_relief_factor` | **0.35** | 主体运动阈值放宽因子 |
| `coverage_motion_threshold` | **0.5** | 运动像素判定阈值（mag > 0.5 视为运动） |
| `temporal_std_threshold` | **0.5** | 时序标准差 sigmoid 中心 |
| `camera_score_floor` | **0.25** | 相机因子得分下限 |

### 5.3 场景类型判定

| 条件 | 类型 |
|------|------|
| `camera_magnitude > 0.5` 且 `static_ratio > 0.5` | static（静态场景） |
| 其他 | dynamic（动态场景） |

### 5.4 动态度等级

| 区间 | 等级 |
|------|------|
| `[0.0, 0.2)` | 极低动态（纯静态） |
| `[0.2, 0.4)` | 低动态 |
| `[0.4, 0.6)` | 中等动态 |
| `[0.6, 0.8)` | 高动态 |
| `[0.8, 1.0]` | 极高动态 |

### 5.5 MLLM 自然度

| 条件 | 分数 |
|------|------|
| MLLM 判定 `is_natural = True` | **1.0** |
| MLLM 判定 `is_natural = False` | **0.3** |

### 5.6 相机补偿

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `_MIN_MATCH_COUNT` | 8 | 最小 SIFT/ORB 匹配点数 |
| `_RANSAC_REPROJ_THRESHOLD` | **3.0** | RANSAC 重投影误差阈值（像素） |

### 5.7 评分公式

```
smoothness = (0.5 × flow_smoothness + 0.5 × trajectory_score) / total  (有轨迹时)
           = flow_smoothness                                          (无轨迹时)

motion_logic_score = clip(
      (0.3 × dynamics + 0.4 × smoothness + 0.3 × naturalness)    (有 MLLM)
    / (0.3 × dynamics + 0.4 × smoothness) / (0.3 + 0.4)          (无 MLLM)
, 0, 1)
```

---

## 6. D5 物理常识一致性

**源码**: `src/physics_consistency/`

### 6.1 配置参数 (`PhysicsConfig`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `drift_flow_threshold` | **0.5** | 像素漂移光流阈值 |
| `drift_min_frames` | 5 | 最小连续漂移帧数 |
| `drift_direction_tolerance` | **30.0** | 漂移方向一致性容差（度数） |
| `enable_mllm` | True | 启用 VLM 判定 |
| `drift_fallback_weight` | 1.0 | 无 VLM 时漂移评分权重 |

### 6.2 评分逻辑

```
drift_score = 1.0                                    (无漂移事件)
            = max(0.0, 1.0 - drift_events × 0.3)     (有漂移事件)

physics_score = VLM 返回的 physics_score (主路径)
              = drift_fallback_weight × drift_score  (VLM 不可用时)
```

---

## 7. D6 背景一致性

**源码**: `src/background_consistency/`

### 7.1 配置参数 (`BackgroundConfig`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `residual_weight` | 0.3 | 残差评分权重 |
| `homography_weight` | 0.3 | 单应性稳定性权重 |
| `depth_weight` | 0.4 | 深度一致性权重 |
| `enable_region_analysis` | False | 是否启用区域分析修正 |

### 7.2 评分公式

```
background_score = clip(
      0.3 × residual_score
    + 0.3 × homography_stability
    + 0.4 × depth_consistency
, 0, 1)

# 当 enable_region_analysis = True 且可用时:
background_score = 0.85 × base_score + 0.15 × region_score
```

---

## 8. 时间一致性 (TCS-lite)

**源码**: `src/temporal_coherence/`

### 8.1 配置参数 (`TemporalCoherenceConfig`)

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

### 8.2 事件分类逻辑

| 事件类型 | 条件 |
|----------|------|
| `edge_emerge` | 从画面边缘出现 |
| `edge_vanish` | 从画面边缘消失 |
| `small_emerge` | 检测框面积增长超过 1.35 倍 |
| `small_vanish` | 检测框面积缩小低于 0.65 倍 |
| `detect_gap` | 轨迹长度不足 min_track_len_steps |
| `abnormal` | 不符合以上条件的出现/消失（真正异常） |

### 8.3 评分公式

```
temporal_coherence_score = 1.0                                        (无事件)
                         = max(0.0, 1.0 - abnormal / all_events)      (有事件)
```

---

## 9. 感知质量

**源码**: `src/perceptual_quality/`

### 9.1 配置参数 (`PerceptualQualityConfig`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `blur_weight` | 0.5 | 模糊度权重 |
| `consistency_weight` | 0.3 | 一致性权重 |
| `artifact_weight` | 0.2 | 瑕疵权重 |
| `laplacian_blur_threshold` | **100.0** | Laplacian 方差 sigmoid 中心 |

### 9.2 模糊等级阈值 (`BlurDetectionConfig`)

| 等级 | 阈值 | 说明 |
|------|------|------|
| mild (轻微模糊) | **0.015** | 轻度模糊判定 |
| moderate (中等模糊) | **0.025** | 中度模糊判定 |
| severe (严重模糊) | **0.04** | 严重模糊判定 |

### 9.3 Laplacian 降级方案

| 参数 | 默认值 | 说明 |
|------|--------|------|
| sigmoid 中心 | 100.0 | 方差中心值 |
| sigmoid 陡度 | 0.02 | 曲线陡峭度 |
| artifact 突变阈值 | **0.3** | 质量突降判定阈值 |

### 9.4 评分公式

```
# Laplacian 方案
blur_score = mean(sigmoid(lap_var))
consistency_score = 1.0 - clip(std(quality_scores) × 2, 0, 1)
artifact_score = 1.0 - mean(diff(quality_scores) > 0.3)

perceptual_quality_score = clip(
      0.5 × blur_score
    + 0.3 × consistency_score
    + 0.2 × artifact_score
, 0, 1)
```

---

## 附录 A. 完整阈值速查表

| 维度 | 参数名 | 默认值 | 用途 |
|------|--------|--------|------|
| D1 | `drop_threshold` | 0.3 | 人脸相似度突降检测 |
| D1 | `match_threshold` | 0.4 | 人脸追踪匹配 |
| D2 | `AU_ACTIVATION_THRESHOLD` | 1.0 | AU 激活判定 |
| D2 | `max_possible_diff` | 5.0 | AU 平滑度归一化 |
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
| PQ | `laplacian_blur_threshold` | 100.0 | Laplacian 模糊 |
| PQ | `blur_thresholds.mild` | 0.015 | 轻微模糊 |
| PQ | `blur_thresholds.moderate` | 0.025 | 中等模糊 |
| PQ | `blur_thresholds.severe` | 0.04 | 严重模糊 |

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
