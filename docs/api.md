# AIGC 视频合理性评测 API 文档

## 概述

AIGC 视频合理性评测 API 提供 RESTful HTTP 接口，支持单视频和批量目录的多维度分析。API 采用异步任务模式：提交任务后立即返回 `job_id`，通过轮询接口获取结果和实时日志。

**适用场景**：业务系统集成、CI/CD 流水线、批量评测脚本。

## 启动服务

```bash
# 安装依赖
pip install -e ".[api]"

# 启动（开发模式）
python scripts/run_api.py --reload

# 启动（生产模式）
python scripts/run_api.py --host 0.0.0.0 --port 8000
```

启动后访问 Swagger 交互文档：`http://localhost:8000/docs`

## 基础信息

- **协议**：HTTP/1.1
- **格式**：JSON（Content-Type: application/json）
- **编码**：UTF-8
- **跨平台**：客户端可在任何平台通过 HTTP 调用

## API 端点

### 1. 健康检查

```
GET /health
```

**响应**：
```json
{"status": "ok"}
```

### 2. 获取配置

```
GET /api/config
```

返回可用的评估维度、范围和支持的 MLLM 提供方。

**响应**：
```json
{
  "scopes": [
    {
      "key": "anomaly",
      "label": "五类异常",
      "description": "身份、表情、生物异常、运动逻辑、物理常识。",
      "dimensions": [
        {"key": "face_identity", "label": "身份一致性", "description": "...", "scope": "anomaly"},
        {"key": "expression", "label": "表情自然度", "description": "...", "scope": "anomaly"},
        ...
      ]
    },
    {
      "key": "full",
      "label": "全量维度",
      "description": "包含时间一致性、背景一致性与感知质量。",
      "dimensions": [...]
    }
  ],
  "mllm_providers": ["vllm", "dashscope", "openai", "anthropic", "huawei_custom"],
  "defaults": {
    "device": "cuda",
    "parallel": true,
    "sample_stride": 2,
    "max_frames": 48,
    "max_side": 640,
    ...
  }
}
```

### 3. 提交分析任务

```
POST /api/evaluate
Content-Type: application/json
```

**请求体参数**：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `video_path` | string | 与 video_dir 二选一 | — | 本地视频文件路径（服务端路径） |
| `video_dir` | string | 与 video_path 二选一 | — | 视频目录路径（批量模式） |
| `scope` | string | 否 | `"anomaly"` | 分析范围：`anomaly`（五类异常）或 `full`（全量维度） |
| `device` | string | 否 | `"cuda"` | 推理设备 |
| `parallel` | bool | 否 | `true` | 是否启用并发检测 |
| `max_workers` | int | 否 | — | 最大并发数（不填则自动） |
| `sample_stride` | int | 否 | `2` | 采样步长（≥1） |
| `max_frames` | int | 否 | `48` | 最大分析帧数（≥2） |
| `max_side` | int | 否 | `640` | 视频最大边长（≥64） |
| `enable_mllm` | bool | 否 | `false` | 是否启用 VLM/MLLM 判定 |
| `mllm_provider` | string | 否 | `"vllm"` | MLLM 提供方 |
| `mllm_model` | string | 否 | — | MLLM 模型名 |
| `mllm_base_url` | string | 否 | — | MLLM API 地址 |
| `mllm_api_key` | string | 否 | — | MLLM API 密钥 |
| `mllm_service_name` | string | 否 | — | huawei_custom 专用 |
| `file_extensions` | string | 否 | `".mp4,.avi,.mov,.mkv,.webm"` | 批量模式扫描扩展名 |
| `recursive_scan` | bool | 否 | `false` | 批量模式是否递归扫描子目录 |

**响应** (202 Accepted)：
```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "queued",
  "video_name": "sample.mp4"
}
```

### 4. 查询任务状态

```
GET /api/jobs/{job_id}
```

**响应**：

```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "running",
  "created_at": 1715600000.0,
  "updated_at": 1715600120.0,
  "completed_at": null,
  "error": null,
  "result_json_path": null,
  "log_path": null,
  "has_result": false,
  "result": null
}
```

状态字段 `status` 取值：

| 值 | 说明 |
|---|---|
| `queued` | 已入队，等待执行 |
| `running` | 正在执行 |
| `completed` | 已完成，`result` 字段包含结果 |
| `failed` | 失败，`error` 字段包含错误信息 |

### 5. 拉取实时日志

```
GET /api/jobs/{job_id}/logs?offset=0
```

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `offset` | int | `0` | 从第几行开始读取 |

**响应**：

```json
{
  "job_id": "a1b2c3d4e5f6",
  "offset": 0,
  "next_offset": 15,
  "lines": [
    "[job] 已创建任务 a1b2c3d4e5f6",
    "[job] 开始分析 data/sample.mp4",
    "[job] 范围=anomaly, 维度=face_identity,expression,...",
    ...
  ],
  "completed": false
}
```

轮询方式：每次请求后用 `next_offset` 作为下次的 `offset`，避免重复拉取。

## 结果格式

### 单视频结果 (`result` 字段)

```json
{
  "video_name": "sample.mp4",
  "final_score": 0.852,
  "elapsed_sec": 45.123,
  "active_dimensions": ["face_identity", "expression", "motion_logic", ...],
  "dimensions": [
    {
      "key": "face_identity",
      "label": "身份一致性",
      "score": 0.910,
      "band": "excellent",
      "metrics": [...],
      "highlights": [...],
      "events": [...],
      "vlm_raw_output": null
    }
  ],
  "summary": {
    "best_dimension": "身份一致性",
    "best_score": 0.910,
    "worst_dimension": "运动逻辑",
    "worst_score": 0.680
  }
}
```

### 批量结果 (`result` 字段)

```json
{
  "batch": true,
  "video_dir": "data/videos",
  "total_videos": 10,
  "completed_videos": 9,
  "failed_videos": 1,
  "elapsed_sec": 480.5,
  "aggregate": {
    "avg_score": 0.723,
    "best_video": "video1.mp4",
    "best_score": 0.892,
    "worst_video": "video3.mp4",
    "worst_score": 0.510
  },
  "video_results": [
    {
      "video_name": "video1.mp4",
      "final_score": 0.892,
      "status": "completed",
      "elapsed_sec": 45.2,
      "active_dimensions": ["face_identity", ...],
      "vlm_outputs": {"motion_logic": {...}},
      "report_path": "outputs/api_results/20260513_video1_report.json"
    }
  ]
}
```

## 调用示例

### curl

```bash
# 健康检查
curl http://localhost:8000/health

# 单视频分析
curl -X POST http://localhost:8000/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{"video_path": "data/sample.mp4", "scope": "anomaly"}'

# 获取结果（替换 job_id）
curl http://localhost:8000/api/jobs/a1b2c3d4e5f6

# 批量分析
curl -X POST http://localhost:8000/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{"video_dir": "data/videos", "scope": "full"}'
```

### Python（标准库，无依赖）

```python
from test_api import APIClient, wait_for_job, print_result

client = APIClient("http://localhost:8000")

# 提交任务
job = client.submit(video_path="data/sample.mp4", scope="anomaly")

# 等待完成（自动打印实时日志）
result = wait_for_job(client, job["job_id"])

# 打印结果
print_result(result["result"])
```

### Python（requests）

```python
import requests
import time

BASE = "http://localhost:8000"

# 提交
resp = requests.post(f"{BASE}/api/evaluate", json={
    "video_path": "data/sample.mp4",
    "scope": "anomaly",
    "enable_mllm": True,
    "mllm_provider": "dashscope",
})
job_id = resp.json()["job_id"]

# 轮询
while True:
    resp = requests.get(f"{BASE}/api/jobs/{job_id}")
    data = resp.json()
    if data["status"] in ("completed", "failed"):
        break
    time.sleep(2)

print(f"综合分: {data['result']['final_score']}")
```

### 测试脚本

项目内置测试脚本 `scripts/test_api.py`：

```bash
python scripts/test_api.py --health
python scripts/test_api.py --video data/sample.mp4
python scripts/test_api.py --dir data/videos
python scripts/test_api.py --video data/sample.mp4 --enable-mllm --mllm-provider dashscope
python scripts/test_api.py --video data/sample.mp4 --url http://10.0.0.5:8000
```

## 错误处理

| HTTP 状态码 | 说明 |
|---|---|
| 200 | 成功 |
| 202 | 任务已接受 |
| 400 | 请求参数错误（响应 `{"error": "..."}` |
| 404 | 任务 ID 不存在 |
| 500 | 服务端异常 |

## 跨平台说明

API 基于标准 HTTP/JSON 协议，客户端可在以下环境运行：

- **Python**：标准库 `urllib` 或 `requests`
- **Shell**：`curl`
- **JavaScript/Node.js**：`fetch`
- **Java**：`HttpURLConnection` / OkHttp
- **Go**：`net/http`
- 任何支持 HTTP 的语言和框架

只需确保客户端能访问 API 服务器的 IP 和端口即可。
