# 视频生成质量与一致性检测维度梳理

> 最后更新: 2026-03-31

---

## 总体进度


| 模块               | 状态                | 测试       | 代码路径                          |
| ---------------- | ----------------- | -------- | ----------------------------- |
| FeatureHub 共享特征层 | **已完成**           | 13 tests | `src/feature_hub/`            |
| MLLM 统一调用层       | **已完成**           | 3 tests  | `src/mllm/`                   |
| D1 人脸身份一致性       | **已完成**           | 9 tests  | `src/face_identity/`          |
| D2 表情自然度         | **已完成**           | 5 tests  | `src/expression_naturalness/` |
| D3 生物特征异常        | **已完成**（三级检测方案）   | 7 tests  | `src/biological_anomaly/`     |
| D4 运动逻辑与平滑度      | **已完成**           | 4 tests  | `src/motion_logic/`           |
| D5 物理常识          | **已完成**           | 2 tests  | `src/physics_consistency/`    |
| D6 环境一致性         | **已完成**           | 4 tests  | `src/background_consistency/` |
| 统一流水线            | **已完成**           | 2 tests  | `src/evaluation_pipeline.py`  |
| **合计**           | **52 tests 全部通过** |          |                               |


---

## 基础设施

### FeatureHub 共享特征层 — **已完成**

- 懒加载 + 内存缓存，避免多维度重复推理（节省 40-60% 推理时间）
- 已实现提取器：光流 (Farneback/RAFT)、人脸嵌入 (InsightFace ArcFace)、深度图 (MiDaS)、主体分割 (Grounding DINO + SAM2)、时序点追踪 (CoTracker + MediaPipe fallback)、瞳孔追踪 (MediaPipe Iris)
- 工厂函数 `create_default_hub(video_path, device)` 一键创建

**下一步：**

- 实现可选磁盘缓存（批量处理场景）
- 设备感知：显存不足时自动卸载已完成模型

### MLLM 统一调用层 — **已完成**

- 支持 OpenAI / Anthropic API 双后端
- hybrid 模式：本地模型优先，失败自动 fallback 到 API
- 已提供 prompt 模板：运动自然度判定、物理常识判定

**下一步：**

- 接入本地开源模型推理（InternVL2-8B / Qwen2-VL-7B）
- 添加 token 用量统计和成本控制

---

## 六个检测维度

### 1. 人脸身份一致性（ID保持） — **已完成**

- **实现方案：** 采用业界标准 **CSIM（余弦相似度身份度量）** 指标
  - InsightFace SCRFD 人脸检测 + ArcFace 512-d 归一化嵌入提取
  - 匈牙利算法跨帧身份关联
  - CSIM 四模式评分：Ref（全局保持）、Adj（时序平滑）、Min（最差情况）、Drop（突变检测）
  - 综合评分：`identity_score = 0.4×Ref + 0.3×Adj + 0.2×Min - 0.1×drop_penalty`
- ~~TI2Net：无可用预训练权重，已弃用~~

**下一步：**

- 可选双骨干：接入 CurricularFace（对表情变化更敏感）
- 多人场景优化：支持多条轨迹独立评分
- 采集实际 AIGC 视频样本进行评分阈值标定

### 2. 表情与肌肉运动自然度 — **已完成**

- **实现方案：**
  - **AU 提取：** Py-Feat 提取 20+ 面部动作单元（AU）强度
  - **AU 组合合理性：** FACS 规则库判定（真笑=AU6+AU12、恐怖谷检测等）+ 冲突对检测
  - **AU 时序平滑度：** 一阶差分标准差评估，自然表情渐进变化 vs AI 伪造突变/锯齿
  - 综合评分：AU 组合得分 × 0.4 + 时序平滑度 × 0.4 + 光流一致性 × 0.2

**下一步：**

- 光流辅助验证：面部区域光流与 AU 强度变化一致性校验
- 丰富 FACS 规则库：补充更多自然表情模式和冲突组合
- 采集样本标定 AU 强度阈值

### 3. 生物特征细节异常 — **已完成**（三级检测 + 口腔已实现）

- **实现方案（三级混合）：**
  - **Level 1 快速筛选：** 基于 EAR/MAR/手部速度 + 全身骨骼
    - 眼：EAR（Eye Aspect Ratio）长时间不眨眼（>90 帧）+ 左右眼 EAR 不对称
    - 嘴：MAR 跳变（嘴型不连续）、持续张嘴超时
    - 手：速度突变、抖动、突然出现/消失等运动模式异常
    - 身体：骨段长度突变、关节角度突变（借鉴 VMBench OIS 滑窗策略）
  - **Level 2 结构检测：** 基于 MediaPipe 关键点的几何/拓扑一致性
    - 嘴：内唇多边形面积突变、嘴部 centroid 跳变（`mouth_area_sudden_change` / `mouth_landmark_jump`）
    - 嘴内颜色：基于内唇 ROI 的 HSV 直方图相关性下降检测（`mouth_color_sudden_change`）
    - 手：手指融合、骨段长度比例异常、关节角度超出范围、手指数量 ≠ 5 等结构异常
  - **Level 3 MLLM 兜底：** 对 L1/L2 提取的疑似异常帧进行 ROI 裁剪，交给 MLLM 做语义判定（可选，按配置启用）
  - **评分：** 三级得分 `level1_score/level2_score/level3_score` 按权重
  `bio_quality_score = 0.3×L1 + 0.4×L2 + 0.3×L3`，并提供按部位正常帧比例（VMBench OIS 风格）

**下一步：**

- 结合实际 AIGC 视频样本标定各类异常的严重度权重（`severity` → penalty）和阈值
- 增强眼部：加入瞳孔固定 / 注视方向异常等规则
- 增强手部：补充自穿透/遮挡下的鲁棒性规则与可视化调试工具
- 优化 Level 3 MLLM prompt：细化异常类别描述，减少误报/漏报

### 4. 运动逻辑与平滑度 — **已完成**

- **实现方案：**
  - **动态度评分：** 光流幅度均值量化运动强度（整合已有 aux_motion_intensity）
  - **平滑度评分：** 光流加速度（二阶导数）检测瞬间跳变，替代效果较差的 Q-Align MSS
  - **运动自然度：** MLLM 辅助判定（可选，需配置 mllm_client）
  - 权重：动态度 × 0.3 + 平滑度 × 0.4 + 自然度 × 0.3（无 MLLM 时自动归一化前两项）

**下一步：**

- 接入轨迹曲率分析（CoTracker 曲率变化率检测物体瞬移）
- 保留 Q-Align MSS 作为补充基线信号
- MLLM 自然度 prompt 优化：扩展异常片段预筛选逻辑

### 5. 物理常识与动力学一致性 — **已完成**

- **实现方案：**
  - **像素漂移检测（高可靠）：** 静态区域光流分析 → 非闭合单向轨迹 = 漂移
  - **重力方向一致性（中等）：** 自由运动轨迹 y 分量抛物线拟合
  - **MLLM 物理判定（实验性）：** 水往高处流、刚体穿模、影子不一致等语义判定
  - 权重：漂移 × 0.5 + 重力 × 0.3 + MLLM × 0.2

**下一步：**

- 重力检测增强：结合深度图判断相机俯仰角补偿
- 利用 CoTracker 轨迹进一步细化自由落体/抛物线拟合与异常模式分类
- MLLM 物理判定 prompt 扩展更多违规场景

### 6. 环境一致性（背景保持） — **已完成**

- **实现方案：**
  - **静态区域残差分析：** 像素残差 + SSIM 局部对比
  - **特征点匹配：** SIFT 特征点 → 单应性矩阵 H 稳定度
  - **深度图时序一致性：** 相邻帧深度图 Pearson 相关系数（闪烁检测）
  - 权重：残差 × 0.3 + 单应性 × 0.3 + 深度一致性 × 0.4

**下一步：**

- 前景/背景分离优化：接入 SAM 分割实现更精准的背景掩码
- 深度边缘与 RGB 边缘对齐度检测
- 空间结构连通性分析（深度断层检测）
- 颜色直方图一致性补充

---

## 统一流水线 — **已完成**

- `EvaluationPipeline.evaluate(video_path)` 一键六维度评测
- 不适用维度自动跳过 + 权重归一化重分配
- 结构化输出 `EvaluationReport`（每维度得分 + 详情 + 最终加权总分）

**默认权重配置：**


| 维度       | 默认权重 | 跳过条件   |
| -------- | ---- | ------ |
| D1 人脸身份  | 0.20 | 无人脸    |
| D2 表情自然度 | 0.15 | 无人脸    |
| D3 生物特征  | 0.10 | 无人脸且无手 |
| D4 运动逻辑  | 0.25 | 全静态    |
| D5 物理常识  | 0.15 | 无运动物体  |
| D6 环境一致性 | 0.15 | 不跳过    |


**下一步：**

- 封装命令行脚本：`python scripts/evaluate.py --input video.mp4 --output report.json`
- 批量评测 + CSV/JSON 报告导出
- 视频类型自动分类（human / scene / object / mixed）
- 可视化报告生成（雷达图 + 异常帧标注）

---

## 技术栈与依赖


| 依赖                     | 用途                | 维度         |
| ---------------------- | ----------------- | ---------- |
| InsightFace            | 人脸检测 + ArcFace 嵌入 | D1         |
| Py-Feat                | AU 强度提取           | D2         |
| MediaPipe              | 关键点检测（手/眼）        | D3         |
| OpenCV                 | 光流 / 特征匹配 / 图像处理  | D4, D5, D6 |
| MiDaS (timm)           | 单目深度估计            | D6         |
| OpenAI / Anthropic SDK | MLLM API 调用       | D4, D5     |
| SciPy                  | 匈牙利算法 / 信号分析      | D1, D4     |


---

## 设计文档

- 详细设计：`docs/superpowers/specs/2026-03-10-six-dimension-evaluation-design.md`
- 实施计划：`docs/superpowers/plans/2026-03-10-six-dimension-evaluation.md`