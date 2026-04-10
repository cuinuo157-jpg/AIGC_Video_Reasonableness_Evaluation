#!/usr/bin/env python3
"""
使用阿里云百炼 / DashScope 多模态 API 对视频做「合理性」理解评测（VLM 裁判）。

依赖:
    pip install dashscope>=1.19.0
    或使用: uv sync --extra dashscope

环境变量:
    DASHSCOPE_API_KEY   必填，百炼 API Key
    DASHSCOPE_BASE_URL  可选，国际区等需设置 endpoint，例如:
        https://dashscope-intl.aliyuncs.com/api/v1

说明:
    - 与 HF Gradio 示例不同，此处走官方 HTTP API（MultiModalConversation）。
    - 默认从本地 mp4 **均匀抽帧** 为 JPG 列表再传入（与 dashscope 官方 samples 一致）。

用法示例:
    python scripts/eval_video_reasonableness_dashscope.py --video data/sample.mp4
    python scripts/eval_video_reasonableness_dashscope.py --video data/clips/ --model qwen3-vl-8b-thinking
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
_repo_root_str = str(REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

from src.mllm.dashscope_video_reasonableness import (
    DEFAULT_SYSTEM_PROMPT,
    build_user_text,
    call_vlm,
    configure_dashscope,
    extract_frame_paths,
    parse_json_from_model_text,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "vlm_reasonableness_dashscope"


def _load_repo_dotenv() -> None:
    """从仓库根 .env 注入环境变量（不覆盖已存在项）。"""
    path = REPO_ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ[key] = val


def _ensure_dashscope():
    try:
        import dashscope
        from dashscope import MultiModalConversation
    except ImportError as e:
        raise ImportError(
            "请先安装: pip install dashscope>=1.19.0 或 uv sync --extra dashscope"
        ) from e
    return dashscope, MultiModalConversation


def collect_videos(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            raise ValueError(f"不支持的文件类型: {path}")
        return [path]
    if path.is_dir():
        vids = sorted(
            p
            for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        )
        if not vids:
            raise ValueError(f"目录下无支持的视频文件: {path}")
        return vids
    raise FileNotFoundError(path)


def run_one_video(
    video_path: Path,
    *,
    model: str,
    api_key: str,
    max_frames: int,
    task_prompt: str,
    context: str | None,
    system_prompt: str,
    stream: bool,
    dashscope: Any,
    MultiModalConversation: Any,
) -> dict[str, Any]:
    frame_paths, tmp_dir = extract_frame_paths(video_path, max_frames=max_frames)
    try:
        t0 = time.perf_counter()
        user_text = build_user_text(task_prompt, context)
        raw = call_vlm(
            model=model,
            api_key=api_key,
            frame_paths=frame_paths,
            system_prompt=system_prompt,
            user_text=user_text,
            stream=stream,
            dashscope=dashscope,
            MultiModalConversation=MultiModalConversation,
        )
        elapsed = time.perf_counter() - t0
        try:
            structured = parse_json_from_model_text(raw)
        except ValueError:
            structured = {
                "parse_error": True,
                "raw_text": raw,
            }
        return {
            "video_path": str(video_path.resolve()),
            "model": model,
            "elapsed_sec": round(elapsed, 3),
            "raw_response": raw,
            "structured": structured,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DashScope 视频合理性 VLM 评测")
    p.add_argument(
        "--video",
        type=Path,
        required=True,
        help="单个视频文件或包含 mp4 等的目录",
    )
    p.add_argument(
        "--model",
        type=str,
        default="qwen3-vl-8b-thinking",
        help="多模态模型名（默认 qwen3-vl-8b-thinking，可按控制台文档替换）",
    )
    p.add_argument("--max-frames", type=int, default=16, help="均匀采样帧数上限")
    p.add_argument(
        "--context",
        type=str,
        default="",
        help="可选：生成意图 / prompt / 业务描述，供模型对照",
    )
    p.add_argument(
        "--task-prompt",
        type=str,
        default="请评估该视频作为 AI 生成或合成内容时，在运动和视觉上是否合理、自然。",
        help="用户任务说明（会附加在帧输入之后）",
    )
    p.add_argument(
        "--system-prompt",
        type=str,
        default="",
        help="覆盖默认 system prompt（留空则使用内置 JSON 输出约束）",
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("DASHSCOPE_API_KEY", ""),
        help="API Key，默认读环境变量 DASHSCOPE_API_KEY",
    )
    p.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("DASHSCOPE_BASE_URL", ""),
        help="可选 API base，如国际区 https://dashscope-intl.aliyuncs.com/api/v1",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="JSON 结果输出目录",
    )
    p.add_argument("--stream", action="store_true", help="流式输出（逐块拼接）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _load_repo_dotenv()
    args = parse_args(argv)
    if not args.api_key.strip():
        print("错误: 请设置环境变量 DASHSCOPE_API_KEY 或使用 --api-key", file=sys.stderr)
        return 2

    dashscope, MultiModalConversation = _ensure_dashscope()
    base = args.base_url.strip() or None
    configure_dashscope(dashscope, base)

    system_prompt = args.system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
    videos = collect_videos(args.video.resolve())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch_id = time.strftime("%Y%m%d_%H%M%S")
    all_results: list[dict[str, Any]] = []

    print("=" * 60)
    print("DashScope 视频合理性评测")
    print(f"模型: {args.model}")
    print(f"视频数: {len(videos)}")
    print(f"输出: {args.output_dir}")
    print("=" * 60)

    for i, vp in enumerate(videos):
        print(f"\n[{i + 1}/{len(videos)}] {vp.name}")
        try:
            row = run_one_video(
                vp,
                model=args.model,
                api_key=args.api_key.strip(),
                max_frames=args.max_frames,
                task_prompt=args.task_prompt,
                context=args.context or None,
                system_prompt=system_prompt,
                stream=args.stream,
                dashscope=dashscope,
                MultiModalConversation=MultiModalConversation,
            )
            all_results.append(row)
            s = row.get("structured")
            if isinstance(s, dict) and not s.get("parse_error"):
                print(f"  overall_score={s.get('overall_score')} is_reasonable={s.get('is_reasonable')}")
            else:
                print("  警告: JSON 解析失败，见结果文件 raw_response")
        except Exception as e:
            print(f"  失败: {e}")
            all_results.append({"video_path": str(vp.resolve()), "error": str(e)})

    out_path = args.output_dir / f"reasonableness_{batch_id}.json"
    payload = {
        "batch_id": batch_id,
        "model": args.model,
        "max_frames": args.max_frames,
        "results": all_results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out_path}")
    return 0 if all("error" not in r for r in all_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
