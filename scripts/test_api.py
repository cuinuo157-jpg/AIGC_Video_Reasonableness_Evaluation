#!/usr/bin/env python3
"""API 调用测试 / 示例脚本。

演示如何通过 HTTP 调用 AIGC 视频合理性评测 API。
纯 HTTP 客户端，无需安装项目依赖，只需 Python 标准库即可运行。

用法:
    # 单视频
    python scripts/test_api.py --video data/sample.mp4

    # 批量目录
    python scripts/test_api.py --dir data/videos

    # 指定 API 地址
    python scripts/test_api.py --video data/sample.mp4 --url http://10.0.0.5:8000

    # 健康检查
    python scripts/test_api.py --health

跨平台: 客户端可在 Windows / Linux / macOS 任意平台运行，
只要能访问 API 服务器的 HTTP 端口即可。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class APIClient:
    """Minimal HTTP client for the evaluation API."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, body: dict | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")
            try:
                err = json.loads(body_text)
                print(f"  API 错误 [{e.code}]: {err.get('error', body_text)}")
            except json.JSONDecodeError:
                print(f"  HTTP 错误 [{e.code}]: {body_text[:200]}")
            raise
        except urllib.error.URLError as e:
            print(f"  连接失败: {e.reason}")
            print(f"  请确认 API 服务已启动: python scripts/run_api.py")
            raise

    def health(self) -> dict:
        return self._request("GET", "/health")

    def config(self) -> dict:
        return self._request("GET", "/api/config")

    def submit(self, **params: Any) -> dict:
        """提交分析任务。可传 video_path 或 video_dir。"""
        return self._request("POST", "/api/evaluate", body=params)

    def job_status(self, job_id: str) -> dict:
        return self._request("GET", f"/api/jobs/{job_id}")

    def job_logs(self, job_id: str, offset: int = 0) -> dict:
        return self._request("GET", f"/api/jobs/{job_id}/logs?offset={offset}")


def wait_for_job(client: APIClient, job_id: str, poll_interval: float = 2.0) -> dict:
    """轮询等待任务完成，实时打印日志。"""
    log_offset = 0
    print(f"\n  任务 {job_id} 已提交，等待结果...")
    print("  " + "-" * 50)

    while True:
        # Pull logs
        try:
            log_data = client.job_logs(job_id, offset=log_offset)
            for line in log_data.get("lines", []):
                print(f"  {line}")
            log_offset = log_data.get("next_offset", log_offset)
        except Exception:
            pass

        # Check status
        status_data = client.job_status(job_id)
        status = status_data.get("status")
        result = status_data.get("result")

        if status in ("completed", "failed"):
            # Final log pull
            try:
                log_data = client.job_logs(job_id, offset=log_offset)
                for line in log_data.get("lines", []):
                    print(f"  {line}")
            except Exception:
                pass
            return status_data

        time.sleep(poll_interval)


def print_result(data: dict) -> None:
    """简洁打印单视频结果。"""
    print("\n" + "=" * 60)
    if data.get("batch"):
        print(f"批量结果: {data.get('video_dir')}")
        print(f"  总数: {data['total_videos']} | 成功: {data['completed_videos']} | 失败: {data['failed_videos']}")
        agg = data.get("aggregate", {})
        print(f"  平均分: {agg.get('avg_score', '-')}")
        print(f"  最优: {agg.get('best_video', '-')} ({agg.get('best_score', '-')})")
        print(f"  最弱: {agg.get('worst_video', '-')} ({agg.get('worst_score', '-')})")
        print(f"\n  视频列表:")
        for i, video in enumerate(data.get("video_results", []), 1):
            score = video.get("final_score")
            score_str = f"{score:.3f}" if score is not None else "-"
            status_icon = "✓" if video["status"] == "completed" else "✗"
            print(f"    {i:2d}. {status_icon} {video['video_name']:<40s} {score_str}")
    else:
        print(f"视频: {data.get('video_name', '-')}")
        print(f"综合分: {data.get('final_score', '-')}")
        print(f"耗时: {data.get('elapsed_sec', '-')}s")
        print(f"活跃维度: {data.get('active_dimensions', [])}")
        summary = data.get("summary", {})
        print(f"最佳: {summary.get('best_dimension', '-')} ({summary.get('best_score', '-')})")
        print(f"最弱: {summary.get('worst_dimension', '-')} ({summary.get('worst_score', '-')})")
        print(f"\n维度详情:")
        for dim in data.get("dimensions", []):
            status = "✓" if dim.get("applicable") else "✗跳过"
            score = f"{dim['score']:.3f}" if dim.get("score") is not None else "-"
            print(f"  {status} {dim['label']:<12s} {score}")
        print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AIGC 视频合理性评测 API 测试客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --health                              # 健康检查
  %(prog)s --video data/sample.mp4                # 单视频分析
  %(prog)s --video data/sample.mp4 --scope full   # 全维度分析
  %(prog)s --dir data/videos                      # 批量目录分析
  %(prog)s --video data/sample.mp4 --enable-mllm --mllm-provider dashscope
        """,
    )
    parser.add_argument("--url", default="http://localhost:8000", help="API 地址 (默认 http://localhost:8000)")
    parser.add_argument("--health", action="store_true", help="健康检查")
    parser.add_argument("--config", action="store_true", help="查看 API 配置")
    parser.add_argument("--video", help="单视频路径")
    parser.add_argument("--dir", help="视频目录（批量模式）")
    parser.add_argument("--scope", default="anomaly", choices=["anomaly", "full"], help="分析范围")
    parser.add_argument("--device", default="cuda", help="推理设备")
    parser.add_argument("--enable-mllm", action="store_true", help="启用 MLLM/VLM")
    parser.add_argument("--mllm-provider", default="huawei_custom", help="MLLM 提供方")
    parser.add_argument("--mllm-model", help="MLLM 模型名")
    parser.add_argument("--parallel", action="store_true", default=True, help="并发检测")
    parser.add_argument("--no-parallel", action="store_false", dest="parallel", help="关闭并发")
    parser.add_argument("--json", action="store_true", help="直接输出原始 JSON（不格式化）")
    args = parser.parse_args()

    client = APIClient(args.url)

    # Health check
    if args.health:
        result = client.health()
        print(f"API 状态: {result['status']}")
        print(f"地址: {args.url}")
        print(f"交互文档: {args.url}/docs")
        return

    # Config
    if args.config:
        cfg = client.config()
        print("API 配置:")
        for scope in cfg.get("scopes", []):
            print(f"  [{scope['key']}] {scope['label']}: {scope['description']}")
            for dim in scope.get("dimensions", []):
                print(f"    - {dim['key']}: {dim['label']}")
        print(f"\n  支持 MLLM: {cfg.get('mllm_providers', [])}")
        return

    # Submit job
    if not args.video and not args.dir:
        print("请指定 --video 或 --dir。使用 --help 查看用法。")
        sys.exit(1)

    if args.dir:
        print(f"批量分析: {args.dir}")
        job_data = client.submit(
            video_dir=args.dir,
            scope=args.scope,
            device=args.device,
            enable_mllm=args.enable_mllm,
            mllm_provider=args.mllm_provider,
            parallel=args.parallel,
            recursive_scan=False,
        )
    else:
        video_path = args.video
        if not Path(video_path).exists():
            print(f"视频不存在: {video_path}")
            sys.exit(1)
        print(f"单视频分析: {video_path}")
        job_data = client.submit(
            video_path=video_path,
            scope=args.scope,
            device=args.device,
            enable_mllm=args.enable_mllm,
            mllm_provider=args.mllm_provider,
            parallel=args.parallel,
        )

    job_id = job_data["job_id"]
    print(f"任务已创建: {job_id}")

    # Wait & print
    result = wait_for_job(client, job_id)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        final_data = result.get("result")
        if final_data:
            print_result(final_data)
        else:
            print(f"\n任务失败: {result.get('error', '未知错误')}")


if __name__ == "__main__":
    main()
