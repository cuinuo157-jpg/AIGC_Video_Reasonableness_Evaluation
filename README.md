# AIGC Video Reasonableness Evaluation

AIGC 生成式视频合理性多维度评测框架。  
围绕 **身份一致性、表情自然度、生物特征异常、运动逻辑、物理常识、背景一致性、时间一致性、感知质量** 等维度进行综合打分与异常定位，支持 FeatureHub 共享特征缓存、可选 MLLM/VLM 判定、统一流水线输出。

---

## README 更新说明（基于 plan.md + 提交记录）

当前 README 已按 `plan.md`（2026-04-09）和近期提交记录同步，主要修正：

- 补齐当前主线模块：`face_identity / expression_naturalness / biological_anomaly / motion_logic / physics_consistency / background_consistency / temporal_coherence / perceptual_quality / evaluation_pipeline`
- 移除历史/不准确描述（如 `scene_realism`、`fusion_engine`、`video_io` 等根目录模块说明）
- 对齐现有脚本入口（`debug_physics.py`、`eval_video_reasonableness_dashscope.py`、`run_4gpu_batch.sh` 等）
- 对齐近期能力更新：D4 动态度脚本参数与输出增强、D5 VLM 判定链路、MLLM 调用日志与抽帧信息输出

---

## 当前能力状态（摘自 plan.md）

| 模块 | 状态 | 代码路径 |
| --- | --- | --- |
| FeatureHub 共享特征层 | 已完成 | `src/feature_hub/` |
| MLLM 统一调用层 | 已完成 | `src/mllm/` |
| D1 人脸身份一致性 | 已完成 | `src/face_identity/` |
| D2 表情自然度 | 已完成 | `src/expression_naturalness/` |
| D3 生物特征异常（三级检测） | 已完成 | `src/biological_anomaly/` |
| D4 运动逻辑与平滑度 | 已完成 | `src/motion_logic/` |
| D5 物理常识一致性（VLM+CoT） | 已完成 | `src/physics_consistency/` |
| D6 背景一致性 | 已完成 | `src/background_consistency/` |
| D7 时间一致性（TCS-lite） | 已完成 | `src/temporal_coherence/` |
| 感知质量 | 已接入流水线 | `src/perceptual_quality/` |
| 统一评测流水线 | 已完成 | `src/evaluation_pipeline.py` |

> 详细进展、设计与下一步计划请查看 `plan.md`。

---

## 目录结构

```text
AIGC_Video_Reasonableness_Evaluation
├─ src/
│  ├─ feature_hub/              # 共享特征层（光流/深度/人脸/分割/追踪）
│  ├─ face_identity/            # D1 人脸身份一致性
│  ├─ expression_naturalness/   # D2 表情自然度
│  ├─ biological_anomaly/       # D3 生物特征异常（三级检测）
│  ├─ motion_logic/             # D4 运动逻辑与平滑度
│  ├─ physics_consistency/      # D5 物理常识一致性
│  ├─ background_consistency/   # D6 背景一致性
│  ├─ temporal_coherence/       # D7 时间一致性（TCS-lite）
│  ├─ perceptual_quality/       # 感知质量
│  ├─ mllm/                     # MLLM/VLM 客户端与提示词
│  └─ evaluation_pipeline.py    # 统一评测入口
├─ scripts/                     # 调试与评测脚本入口
├─ tests/                       # 单元测试
├─ data/                        # 示例数据（自备）
├─ outputs/                     # 输出目录
├─ third_party/                 # 第三方模型与依赖
├─ plan.md                      # 当前阶段计划与进展
└─ README.md
```

---

## 环境准备

推荐 Python 3.10+。

### 方式一：使用 uv（推荐）

```bash
uv sync
```

若需要 DashScope 视频评测能力：

```bash
uv sync --extra dashscope
```

### 方式二：使用 pip

```bash
pip install -r requirements.txt
```

依赖兼容说明：

- 当前项目依赖 `mediapipe` 与 `py-feat/nltools`，两者共同约束要求 `numpy < 1.24`
- 因此项目已将 `numpy` 固定在 `>=1.23.5,<1.24`
- 如果你的环境里已经装成了 `numpy 2.x`，需要先降级再安装项目依赖

### 环境变量

复制并编辑 `.env.example`：

```bash
cp .env.example .env
```

常用变量：

- `DASHSCOPE_API_KEY`：DashScope 视频 VLM 所需
- `DASHSCOPE_BASE_URL`：可选（国际区等）
- `VLLM_OPENAI_BASE_URL` / `VLLM_API_KEY`：本地 OpenAI 兼容 VLLM 服务

ONNX Runtime 说明：

- 项目默认依赖 `onnxruntime`，用于 `insightface` 等 ONNX 模型推理
- 如果你使用 GPU 版 ONNX Runtime，可手动将 `onnxruntime` 替换为 `onnxruntime-gpu`
- 不建议在同一环境里同时安装 `onnxruntime` 和 `onnxruntime-gpu`

---

## 快速开始（按模块调试）

> 以下命令均为当前仓库已存在脚本。

### 1) D4 运动逻辑

```bash
python scripts/debug_dynamics.py \
  --input data/sample.mp4 \
  --device cuda \
  --method raft \
  --subject \
  --save-vis
```

可选 `--enable-mllm` 启用 MLLM 判定。近期已增强抽帧信息、耗时统计、MLLM 输入输出日志。

### 2) D5 物理常识一致性

```bash
python scripts/debug_physics.py \
  --input data/sample.mp4 \
  --device cuda \
  --enable-mllm
```

默认可走本地 OpenAI 兼容 VLLM；也可通过 `--mllm-provider dashscope` 切换百炼。

### 3) D3 生物特征异常

```bash
python scripts/debug_bio_anomaly.py \
  --input data/sample.mp4 \
  --device cuda \
  --save-vis
```

### 4) D7 时间一致性（TCS-lite）

```bash
python scripts/debug_temporal_coherence.py \
  --input data/sample.mp4 \
  --device cuda \
  --save-det-vis
```

### 5) 其他模块

```bash
python scripts/debug_expression.py --input data/sample.mp4 --device cuda
python scripts/debug_face_identity.py --input data/sample.mp4 --device cuda
python scripts/debug_iris_tracking.py --input data/sample.mp4 --device cuda --save-vis
```

### 6) DashScope 视频合理性评测脚本

```bash
python scripts/eval_video_reasonableness_dashscope.py \
  --video data/sample.mp4 \
  --model qwen3-vl-8b-thinking
```

### 7) Web 可视化界面

```bash
python scripts/run_webui.py --host 127.0.0.1 --port 8080
```

打开 `http://127.0.0.1:8080`，可通过页面：

- 上传视频或填写本地视频路径
- 选择“五类异常”或“全量维度”
- 配置 `sample_stride / max_frames / max_side / parallel`
- 查看综合分、维度分、异常摘要与事件列表

---

## 统一流水线（代码调用）

统一入口为 `src/evaluation_pipeline.py` 的 `EvaluationPipeline.evaluate(video_path)`。

示例：

```python
from src.evaluation_pipeline import EvaluationPipeline

pipeline = EvaluationPipeline(device="cuda", enable_mllm=False)
report = pipeline.evaluate("data/sample.mp4")

print(report.final_score)
print(report.active_dimensions)
```

说明：

- 默认会对可用维度进行评测，并对不可用维度做跳过处理
- 对参与维度执行权重归一化后得到 `final_score`
- 各维度详细结果保存在 `report.dimensions`

---

## 批量执行（Dynamics 多卡）

项目提供 `scripts/run_4gpu_batch.sh`（名称固定，GPU 数量可配置）：

```bash
bash scripts/run_4gpu_batch.sh \
  --input-dir /data/videos \
  --gpus 0,1,2,3 \
  --method raft \
  --subject \
  --offline \
  --save-vis
```

输出位于 `outputs/dynamics_batch/<timestamp>/`。

---

## 开发与测试

```bash
pytest tests/ -v --tb=short
ruff check src/ scripts/
ruff format src/ scripts/
```

---

## 相关文档

- 进展计划：`plan.md`
- 源码说明：`src/README.md`
- 脚本说明：`scripts/README.md`
- 历史统一流水线文档：`scripts/unified_pipeline_README.md`（已标注历史状态）

---

## 致谢

项目集成/参考了多个开源能力，详见 `third_party/` 目录与各自许可证。常见依赖包括：

- Grounded-SAM-2
- Co-Tracker
- GroundingDINO
- RAFT
- Q-Align
- InsightFace / MediaPipe / Py-Feat 等
