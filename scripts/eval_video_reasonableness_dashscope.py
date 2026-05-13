#!/usr/bin/env python3
"""
使用阿里云百炼 / DashScope 多模态 API 对视频做「合理性」理解评测（VLM 裁判）。

依赖:
    pip install dashscope>=1.19.0
    或使用: uv sync --extra dashscope

配置方式（优先级从高到低）:
    1. CLI 参数: --api-key, --model, --base-url
    2. .env 环境变量: DASHSCOPE_API_KEY, MLLM_PROVIDER, MLLM_MODEL 等（通过 MLLMConfig 读取）
    3. 硬编码兜底值（仅 model 有: qwen3-vl-8b-thinking）

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
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
_repo_root_str = str(REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

from src.mllm.config import MLLMConfig
from src.mllm.dashscope_video_reasonableness import (
    DEFAULT_SYSTEM_PROMPT,
    build_user_text,
    call_vlm,
    configure_dashscope,
    extract_frame_paths,
    parse_json_from_model_text,
)
from src.mllm.dotenv_loader import load_dotenv

DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "vlm_reasonableness_dashscope"



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
        default="",
        help="多模态模型名（留空则从环境变量 MLLM_MODEL 读取，兜底 qwen3-vl-8b-thinking）",
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
        default="",
        help="API Key（留空则从 MLLM 配置 / 环境变量读取）",
    )
    p.add_argument(
        "--base-url",
        type=str,
        default="",
        help="可选 API base（留空则从 MLLM 配置 / 环境变量读取），如国际区 https://dashscope-intl.aliyuncs.com/api/v1",
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
    load_dotenv()
    cfg = MLLMConfig.from_env()
    args = parse_args(argv)

    # ── 合并配置：CLI 显式参数 > 环境变量 > 硬编码兜底 ──
    api_key = args.api_key.strip() or cfg.api_key or ""
    if not api_key:
        print("错误: 请设置 DASHSCOPE_API_KEY / MLLM_API_KEY 环境变量，或使用 --api-key", file=sys.stderr)
        return 2
    model = args.model.strip() or cfg.api_model or "qwen3-vl-8b-thinking"
    base_url = args.base_url.strip() or cfg.api_base_url or None

    dashscope, MultiModalConversation = _ensure_dashscope()
    configure_dashscope(dashscope, base_url)

    system_prompt = args.system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
    videos = collect_videos(args.video.resolve())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch_id = time.strftime("%Y%m%d_%H%M%S")
    all_results: list[dict[str, Any]] = []

    print("=" * 60)
    print("DashScope 视频合理性评测")
    print(f"模型: {model}")
    print(f"视频数: {len(videos)}")
    print(f"输出: {args.output_dir}")
    print("=" * 60)

    for i, vp in enumerate(videos):
        print(f"\n[{i + 1}/{len(videos)}] {vp.name}")
        try:
            row = run_one_video(
                vp,
                model=model,
                api_key=api_key,
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
        "model": model,
        "max_frames": args.max_frames,
        "results": all_results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out_path}")
    return 0 if all("error" not in r for r in all_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
