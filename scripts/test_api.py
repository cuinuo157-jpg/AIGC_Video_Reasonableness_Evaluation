#!/usr/bin/env python3
"""API 调用测试 / 示例脚本。

演示如何通过 HTTP 调用 AIGC 视频合理性评测 API。
纯 HTTP 客户端，无需安装项目依赖，只需 Python 标准库即可运行。

用法:
    # 单视频
    python test_api.py --video data/sample.mp4

    # 批量目录
    python scripts/test_api.py --dir data/videos

    # 指定 API 地址
    python scripts/test_api.py --video data/sample.mp4 --url http://10.0.0.5:8000

    # 健康检查
    python test_api.py --health

跨平台: 客户端可在 Windows / Linux / macOS 任意平台运行，
只要能访问 API 服务器的 HTTP 端口即可。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _encode_multipart(fields: dict[str, str], file_field: str, file_path: str) -> tuple[bytes, str]:
    """Build multipart/form-data body for file upload. Returns (body, content_type)."""
    boundary = "---APIClientBoundary" + os.urandom(16).hex()
    body_parts: list[bytes] = []
    for name, value in fields.items():
        body_parts.append(f"--{boundary}\r\n".encode())
        body_parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body_parts.append(f"{value}\r\n".encode())
    filename = Path(file_path).name
    body_parts.append(f"--{boundary}\r\n".encode())
    body_parts.append(f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode())
    body_parts.append(b"Content-Type: video/mp4\r\n\r\n")
    body_parts.append(Path(file_path).read_bytes())
    body_parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(body_parts), f"multipart/form-data; boundary={boundary}"


class APIClient:
    """Minimal HTTP client for the evaluation API."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, body: dict | None = None,
                 raw_body: bytes | None = None, content_type: str | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = raw_body if raw_body is not None else (json.dumps(body).encode("utf-8") if body else None)
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", content_type or "application/json")
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

    def submit_upload(self, file_path: str, **params: Any) -> dict:
        """提交分析任务 + 上传视频文件。"""
        fields: dict[str, str] = {}
        for k, v in params.items():
            if v is not None and v != "":
                fields[k] = str(v).lower() if isinstance(v, bool) else str(v)
        raw_body, content_type = _encode_multipart(fields, "file", file_path)
        return self._request("POST", "/api/evaluate/upload", raw_body=raw_body, content_type=content_type)

    def job_status(self, job_id: str) -> dict:
        return self._request("GET", f"/api/jobs/{job_id}")

    def job_logs(self, job_id: str, offset: int = 0) -> dict:
        return self._request("GET", f"/api/jobs/{job_id}/logs?offset={offset}")

    def cleanup(self, job_id: str) -> dict:
        """Delete all server-side temporary files for a completed job."""
        return self._request("POST", f"/api/jobs/{job_id}/cleanup")

    def download_artifacts(self, job_id: str, save_path: str) -> str:
        """Download all generated files as a zip archive. Returns local path."""
        url = f"{self.base_url}/api/jobs/{job_id}/artifacts"
        req = urllib.request.Request(url, method="GET")
        req.add_header("Accept", "application/zip")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
                Path(save_path).write_bytes(data)
                return save_path
        except urllib.error.HTTPError as e:
            print(f"  产物下载失败 [{e.code}]: 任务可能无产物或未完成")
            return ""


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
    parser.add_argument("--scope", default="anomaly", choices=["anomaly", "top5", "full"], help="分析范围")
    parser.add_argument("--device", default="cuda", help="推理设备")
    parser.add_argument("--enable-mllm", action="store_true", help="启用 MLLM/VLM")
    parser.add_argument("--mllm-provider", default="huawei_custom", help="MLLM 提供方")
    parser.add_argument("--mllm-model", default=None, help="MLLM 模型名")
    parser.add_argument("--mllm-base-url", default=None, help="MLLM API 地址")
    parser.add_argument("--mllm-api-key", default=None, help="MLLM API 密钥")
    parser.add_argument("--mllm-service-name", default=None, help="huawei_custom service_name")
    parser.add_argument("--sample-stride", type=int, default=2, help="采样步长")
    parser.add_argument("--max-frames", type=int, default=None, help="最大帧数")
    parser.add_argument("--max-side", type=int, default=None, help="最大边长")
    parser.add_argument("--save-vis", action="store_true", help="生成可视化产物")
    parser.add_argument("--parallel", action="store_true", default=True, help="并发检测")
    parser.add_argument("--no-parallel", action="store_false", dest="parallel", help="关闭并发")
    parser.add_argument("--upload", action="store_true", help="上传本地视频文件到服务器（远程调用时必须）")
    parser.add_argument("--output", default=None, help="本地结果保存路径（默认保存到视频同目录 .result.json）")
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
        need_upload = args.upload or not args.url.startswith("http://localhost") and not args.url.startswith("http://127.")
        if need_upload:
            if not Path(video_path).exists():
                print(f"本地视频不存在: {video_path}")
                sys.exit(1)
            print(f"上传+分析: {video_path} → {args.url}")
            job_data = client.submit_upload(
                file_path=video_path,
                scope=args.scope,
                device=args.device,
                enable_mllm=args.enable_mllm,
                mllm_provider=args.mllm_provider,
                mllm_model=args.mllm_model,
                mllm_base_url=args.mllm_base_url,
                mllm_api_key=args.mllm_api_key,
                mllm_service_name=args.mllm_service_name,
                save_visualizations=args.save_vis,
                parallel=args.parallel,
                sample_stride=args.sample_stride,
                max_frames=args.max_frames,
                max_side=args.max_side,
            )
        else:
            if not Path(video_path).exists():
                print(f"本地视频不存在: {video_path}")
                print(f"  提示: 远程调用请加 --upload 上传文件到服务器")
                sys.exit(1)
            print(f"单视频分析 (服务端路径): {video_path}")
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

    # Save result locally
    result_path = None
    if args.output:
        result_path = Path(args.output)
    elif args.video:
        result_path = Path(args.video).with_suffix(".result.json")
    if result_path:
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已保存到: {result_path}")

    # Download visualization artifacts
    try:
        artifact_zip = Path(args.output or (Path(args.video).stem + "_artifacts")).with_suffix(".zip")
        saved = client.download_artifacts(job_id, str(artifact_zip))
        if saved:
            print(f"可视化产物已保存到: {saved}")
    except Exception:
        pass

    # Clean up server-side temporary files
    try:
        deleted = client.cleanup(job_id)
        if deleted.get("count", 0) > 0:
            print(f"服务端已清理 {deleted['count']} 个临时文件")
    except Exception:
        pass


if __name__ == "__main__":
    main()
