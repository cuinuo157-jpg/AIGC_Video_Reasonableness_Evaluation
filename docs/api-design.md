# AIGC 视频合理性评测 API — 技术设计文档

## 1. 技术路线

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                     客户端层                              │
│  WebUI (stdlib HTTP)  │  FastAPI Docs  │  curl / SDK    │
└──────────────┬──────────────────────────────────────────┘
               │  HTTP/JSON  (POST /api/evaluate, GET /api/jobs/{id})
┌──────────────▼──────────────────────────────────────────┐
│                    接口层 (src/api/)                      │
│  server.py   — FastAPI 路由、CORS、异常处理              │
│  models.py   — Pydantic 请求/响应 Schema（类型校验）      │
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│                  核心逻辑层 (src/api/core.py)             │
│  JobManager      — 异步任务生命周期管理                   │
│  AnalysisConfig  — 不可变配置（frozen dataclass）         │
│  Job             — 任务状态机（queued→running→completed/failed）│
│  run_analysis()  — 分析执行入口                           │
│  parse_analysis_config() — 请求参数解析                   │
│  build_dashboard_report() / build_batch_report() — 报告生成│
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│                  评测引擎层 (src/evaluation_pipeline.py)   │
│  EvaluationPipeline  — 8 维度分析器调度                   │
│  FeatureHub          — 共享特征缓存（光流、深度、人脸等） │
│  ThreadPoolExecutor  — 维度级并发                         │
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│                  维度分析器层 (src/*/analyzer.py)          │
│  FaceIdentity / Expression / Biological / Motion /       │
│  Physics / Background / Temporal / Perceptual            │
└─────────────────────────────────────────────────────────┘
```

### 1.2 技术栈

| 层次 | 技术选型 | 选型理由 |
|---|---|---|
| HTTP 框架 | FastAPI + uvicorn | 自动 OpenAPI 文档、类型校验、高性能 ASGI |
| 数据校验 | Pydantic (v1/v2 兼容) | 请求参数自动校验，减少样板代码 |
| 任务调度 | 标准库 `threading` + daemon 线程 | 零外部依赖，适合 GPU 密集型（非 IO 密集） |
| 日志流 | 内存环形缓冲 + 偏移量轮询 | 简单可靠，无需 WebSocket 或消息队列 |
| 状态管理 | `threading.RLock` + 内存字典 | 单进程内任务管理，重入锁防死锁 |
| 评测引擎 | `ThreadPoolExecutor` | 8 维度并发分析，共享 FeatureHub |
| 报告生成 | Python dataclass + JSON | 结构化输出，易于下游解析 |

### 1.3 设计原则

**零额外依赖**：核心逻辑（`src/api/core.py`）只依赖标准库 + 项目已有依赖，不引入新的第三方包。FastAPI 本身作为可选依赖（`[api]` extra），WebUI 不依赖它。

**单进程多线程**：GPU 推理场景不适合多进程（显存竞争），采用单进程 + daemon 线程执行任务 + HTTP 线程响应请求。批量模式复用同一个 `EvaluationPipeline`实例避免重复加载模型。

**异步非阻塞**：提交任务立即返回 `job_id`（202 Accepted），客户端轮询 `GET /api/jobs/{id}` 获取结果。灵感来自 AWS Batch / SageMaker 异步推理模式。

---

## 2. 核心设计

### 2.1 任务生命周期

```
create_job()                     _run_job() 线程启动
    │                                  │
    ▼                                  ▼
┌────────┐   daemon thread    ┌─────────┐   完成/失败   ┌───────────┐
│ queued │ ────────────────── │ running │ ──────────── │ completed │
└────────┘                    └─────────┘              │  failed   │
                                  │                    └───────────┘
                                  │ 批量模式
                                  ▼
                           批量进度实时更新 job.result.batch_progress
                           (current / total / current_video)
```

**状态转换规则**：
- `queued` → `running`：daemon 线程启动，不可逆
- `running` → `completed`：所有维度分析完成
- `running` → `failed`：任何未捕获异常
- 安全网：`get_job_snapshot()` 和 `get_job_logs()` 检查批量结果数据自推断完成状态，防止 daemon 线程卡死后状态僵死

### 2.2 批量模型复用

批量模式下 `EvaluationPipeline` 在循环外创建一次，避免每个视频重复加载 8 个分析器模型（InsightFace、MediaPipe、RAFT、Py-Feat 等）：

```
批量循环前:
  pipeline = EvaluationPipeline(...)  ← 8 个模型加载到 GPU（一次性）

批量循环内 (每个视频):
  pipeline.evaluate(video_path)
    ├─ FeatureHub(video_path)  ← 读视频、抽帧、提取光流/深度/人脸特征
    ├─ analyzer_1.analyze(hub) ← 复用已加载模型
    ├─ analyzer_2.analyze(hub)
    └─ ...
  torch.cuda.empty_cache()     ← 释放视频特征显存，保留模型
```

每个视频结束后执行 `torch.cuda.empty_cache()` + `gc.collect()` 清理特征图显存但不卸载模型。

### 2.3 日志流设计

不使用 WebSocket 或 SSE，采用**偏移量轮询**：

```
客户端                        服务端
  │                             │
  │ GET /jobs/{id}/logs?offset=0│
  │ ──────────────────────────► │ 返回 [line_0 ... line_N], next_offset=N+1
  │                             │
  │ GET /jobs/{id}/logs?offset=N+1
  │ ──────────────────────────► │ 返回 [line_N+1 ...], next_offset=...
```

**优点**：
- 纯 HTTP，客户端无需 WebSocket 支持
- 断线重连不丢数据（客户端记住 `offset`）
- 服务端无状态，内存只存最近 800 行（环形缓冲）

**日志捕获**：批量模式每次视频分析用 `redirect_stdout` + `_JobLogHandler` 把 Python `print()` 和 `logging` 全部重定向到 job 日志缓冲。

### 2.4 线程安全

| 资源 | 保护方式 |
|---|---|
| `_jobs` 字典 | `threading.RLock()`（重入锁，允许同线程嵌套加锁） |
| `job.logs` 列表 | 与 `_jobs` 同锁保护 |
| `job.status` / `job.result` | 同锁保护，批量完成时在锁内原子更新 |
| `sys.stdout` / `sys.stderr` | `redirect_stdout` context manager，线程局部 |

`RLock` 的选择原因：`_append_log()` 可能在 `with self._lock:` 块内被调用，普通 `Lock` 会死锁。

### 2.5 单视频 vs 批量模式路由

```
parse_analysis_config(payload)
    ├─ video_path 非空 → 单视频模式
    │     └─ _run_job → run_analysis() → build_dashboard_report()
    │
    └─ video_dir 非空 → 批量模式
          └─ _run_job → scan_video_directory() → _run_batch_job()
                └─ 循环: pipeline.evaluate() → build_batch_report()
```

两种模式共用同一个 `_run_job` 入口，通过 `video_dir` 字段区分路径。

---

## 3. 接口设计

### 3.1 RESTful 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 健康检查，返回 `{"status":"ok"}` |
| `GET` | `/api/config` | 获取维度目录、默认配置、支持的 MLLM |
| `POST` | `/api/evaluate` | 提交分析任务（单视频或批量），返回 `job_id` |
| `GET` | `/api/jobs/{id}` | 查询任务状态和结果 |
| `GET` | `/api/jobs/{id}/logs` | 拉取实时日志（支持 offset 增量） |

### 3.2 批量结果数据模型

```python
{
    "batch": True,
    "video_dir": "data/videos",
    "results_dir": "outputs/webui_results",
    "total_videos": 10,
    "completed_videos": 9,
    "failed_videos": 1,
    "aggregate": {
        "avg_score": 0.723,
        "best_video": "...", "best_score": 0.892,
        "worst_video": "...", "worst_score": 0.510
    },
    "video_results": [
        {
            "video_name": "video1.mp4",
            "final_score": 0.892,
            "status": "completed",
            "elapsed_sec": 45.2,
            "active_dimensions": ["face_identity", ...],
            "vlm_outputs": {"motion_logic": {...}},
            "report_path": "outputs/webui_results/..._per_video.json"
        }
    ]
}
```

### 3.3 错误处理分层

```
Pydantic 校验层 (server.py)
    ↓ 400 Bad Request
ValueError / KeyError 处理 (server.py 全局 handler)
    ↓ 400 / 404
_run_job 顶层 try/except (core.py)
    ↓ job.status = "failed"
任务安全网 (get_job_snapshot / get_job_logs)
    ↓ 数据推断已完成状态
```

---

## 4. 部署说明

### 4.1 安装

```bash
pip install -e ".[api]"
```

### 4.2 启动

```bash
# 开发模式（热重载）
python scripts/run_api.py --reload

# 生产模式（多 worker）
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4.3 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `DASHSCOPE_API_KEY` | 阿里云 DashScope API 密钥 | — |
| `VLLM_OPENAI_BASE_URL` | 本地 vLLM 服务地址 | `http://localhost:8201/v1` |
| `MLLM_PROVIDER` | 默认 MLLM 提供方 | `vllm` |
| `MLLM_MODEL` | 默认模型名 | — |

### 4.4 跨平台

API 基于标准 HTTP/JSON，客户端无平台限制。项目提供的 `scripts/test_api.py` 使用纯 Python 标准库，可在 Windows / Linux / macOS 任意平台运行，只需网络能通 API 服务器。
