#!/usr/bin/env python3
"""
D5 物理常识与动力学一致性模块调试脚本。

启用 --enable-mllm 时默认走 huawei_custom 图片帧接口；
也可通过 --mllm-provider 切换到 vllm / dashscope / openai / anthropic。

依赖:
    VLLM: 无额外 Python 包（openai 客户端）
    DashScope: pip install dashscope>=1.19.0

环境变量:
    MLLM_API_BASE_URL     可配置自定义接口地址
    MLLM_API_KEY          可配置自定义接口鉴权
    MLLM_API_SERVICE_NAME 自定义 service_name
    VLLM_OPENAI_BASE_URL  vllm 时使用
    DASHSCOPE_API_KEY     dashscope 时使用

用法示例:
    python scripts/debug_physics.py --input data/sample.mp4
    python scripts/debug_physics.py --input data/sample.mp4 --enable-mllm --save-json outputs/physics_result.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_repo_root_str = str(REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

from src.feature_hub.hub import create_default_hub
from src.physics_consistency.analyzer import PhysicsConsistencyAnalyzer
from src.physics_consistency.config import PhysicsConfig
from src.mllm.config import MLLMConfig
from src.mllm.client import MLLMClient
from src.mllm.dotenv_loader import load_dotenv
from src.mllm.prompts.physics_commonsense import build_physics_prompt

DEFAULT_MLLM_PROVIDER = "huawei_custom"
DEFAULT_MLLM_MODEL = "Qwen3-VL-32B-Instruct"
DEFAULT_MLLM_BASE_URL = "http://aitest-beta.rnd.huawei.com/v1"
DEFAULT_MLLM_SERVICE_NAME = "simple_client"



def _preview_and_save_mllm_frames(
    video_path: str,
    sample_fps: int,
    max_frames: int,
    save_dir: Path,
    label: str = "mllm",
) -> dict:
    """抽取将发送给 MLLM 的帧，打印抽帧统计，并将帧保存到磁盘。"""
    import cv2
    from src.mllm.vllm_openai_video import extract_frames_jpeg_bytes, subsample_uniform

    cap = cv2.VideoCapture(video_path)
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration_sec = total_frames / video_fps if video_fps > 0 else 0
    frame_interval = max(1, int(video_fps / sample_fps))

    raw = extract_frames_jpeg_bytes(video_path, sample_fps)
    sampled = subsample_uniform(raw, max_frames)

    summary = {
        "video_fps": round(video_fps, 2),
        "video_resolution": f"{width}x{height}",
        "video_total_frames": total_frames,
        "video_duration_sec": round(duration_sec, 2),
        "sample_fps": sample_fps,
        "frame_interval": frame_interval,
        "after_fps_sampling": len(raw),
        "after_subsample": len(sampled),
        "max_frames_limit": max_frames,
    }

    print(f"\n{'='*60}")
    print(f"[抽帧情况: {label}]")
    print(f"{'='*60}")
    print(f"  视频: {video_path}")
    print(f"  原始: {width}x{height}, fps={video_fps:.1f}, 总帧={total_frames}, 时长={duration_sec:.1f}s")
    print(f"  step=max(1,int({video_fps:.1f}/{sample_fps}))={frame_interval}, 按 {sample_fps}fps 抽取 → {len(raw)} 帧")
    print(f"  subsample_uniform(max={max_frames}) → 最终送入 {label}: {len(sampled)} 帧")

    save_dir.mkdir(parents=True, exist_ok=True)
    for i, frame_bytes in enumerate(sampled):
        out_path = save_dir / f"frame_{i+1:02d}_of_{len(sampled)}.jpg"
        out_path.write_bytes(frame_bytes)
    print(f"  已保存 {len(sampled)} 张帧到: {save_dir}")

    return summary


def build_mllm_client(args: argparse.Namespace) -> MLLMClient | None:
    """构建 MLLM 客户端（与 debug_dynamics 参数语义一致）。"""
    if not args.enable_mllm:
        return None
    api_key = (args.mllm_api_key or "").strip()
    if args.mllm_provider in {"openai", "anthropic", "dashscope"} and not api_key:
        print(
            "错误: 启用 --enable-mllm 且 provider 为 openai/anthropic/dashscope 时，"
            "必须提供 --mllm-api-key 或设置对应环境变量",
            file=sys.stderr,
        )
        return None
    base_url = (args.mllm_base_url or "").strip()
    if args.mllm_provider == "huawei_custom" and not base_url:
        base_url = DEFAULT_MLLM_BASE_URL
    cfg = MLLMConfig(
        backend="api",
        api_provider=args.mllm_provider,
        api_model=args.mllm_model,
        api_key=api_key or None,
        api_base_url=base_url or None,
        api_service_name=(args.mllm_service_name or "").strip() or DEFAULT_MLLM_SERVICE_NAME,
        dashscope_video_fps=args.mllm_fps,
    )
    return MLLMClient(cfg)


def analyze_video(
    video_path: str,
    device: str = "cuda",
    enable_mllm: bool = False,
    mllm_client: MLLMClient | None = None,
) -> dict:
    """分析视频的物理常识一致性。"""
    print(f"加载视频: {video_path}")
    hub = create_default_hub(video_path, device)

    config = PhysicsConfig(enable_mllm=enable_mllm)
    analyzer = PhysicsConsistencyAnalyzer(config=config, mllm_client=mllm_client)

    print("运行 D5 物理常识分析...")
    result = analyzer.analyze(hub)

    return {
        "applicable": result.applicable,
        "skip_reason": result.skip_reason,
        "physics_score": result.physics_score,
        "drift_events": result.drift_events,
        "drift_score": result.drift_score,
        "vlm_score": result.vlm_score,
        "vlm_reasoning": result.vlm_reasoning,
        "vlm_violations": result.vlm_violations,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D5 物理常识与动力学一致性分析")
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入视频文件路径",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="计算设备 (cuda/cpu)",
    )
    p.add_argument(
        "--enable-mllm",
        action="store_true",
        help=f"启用 VLM 物理判定（默认 {DEFAULT_MLLM_PROVIDER}）",
    )
    p.add_argument(
        "--mllm-provider",
        type=str,
        default=os.environ.get("MLLM_PROVIDER", DEFAULT_MLLM_PROVIDER),
        choices=["vllm", "openai", "anthropic", "dashscope", "huawei_custom"],
        help=f"MLLM 提供方（默认 {DEFAULT_MLLM_PROVIDER}；可通过 MLLM_PROVIDER 环境变量配置）",
    )
    p.add_argument(
        "--mllm-model",
        type=str,
        default=os.environ.get("MLLM_MODEL", DEFAULT_MLLM_MODEL),
        help=f"模型名（默认 {DEFAULT_MLLM_MODEL}；通过 MLLM_MODEL 环境变量配置）",
    )
    p.add_argument(
        "--mllm-api-key",
        type=str,
        default=os.environ.get("MLLM_API_KEY", "")
        or os.environ.get("DASHSCOPE_API_KEY", "")
        or os.environ.get("VLLM_API_KEY", ""),
        help="API Key（openai/anthropic/dashscope 通常必填；vllm/huawei_custom 可空）",
    )
    p.add_argument(
        "--mllm-base-url",
        type=str,
        default=os.environ.get("MLLM_API_BASE_URL", "")
        or os.environ.get("DASHSCOPE_BASE_URL", "")
        or os.environ.get("VLLM_OPENAI_BASE_URL", ""),
        help=f"Base URL（huawei_custom 默认 {DEFAULT_MLLM_BASE_URL}；vllm 默认代码内 localhost:8201/v1）",
    )
    p.add_argument(
        "--mllm-service-name",
        type=str,
        default=os.environ.get("MLLM_API_SERVICE_NAME", DEFAULT_MLLM_SERVICE_NAME),
        help=f"自定义 API 的 service_name（默认 {DEFAULT_MLLM_SERVICE_NAME}；仅 huawei_custom 需要）",
    )
    p.add_argument(
        "--mllm-fps",
        type=int,
        default=int(os.environ.get("MLLM_FPS", "2")),
        help="judge_video_path 抽帧 fps（通过 MLLM_FPS 环境变量配置）",
    )
    p.add_argument(
        "--save-json",
        type=str,
        default="",
        help="可选：保存结果为 JSON 文件",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)

    if not Path(args.input).is_file():
        print(f"错误: 视频文件不存在: {args.input}", file=sys.stderr)
        return 1

    mllm_client = None
    if args.enable_mllm:
        mllm_client = build_mllm_client(args)
        if not mllm_client:
            print("错误: 无法初始化 MLLM 客户端", file=sys.stderr)
            return 1

    if args.enable_mllm:
        stem = Path(args.input).stem
        frames_dir = (
            Path(args.save_json).parent if args.save_json else REPO_ROOT / "outputs" / "physics"
        ) / f"{stem}_mllm_frames"
        _preview_and_save_mllm_frames(
            video_path=args.input,
            sample_fps=args.mllm_fps,
            max_frames=5,
            save_dir=frames_dir,
            label="D5 物理常识 VLM",
        )

    t_total = time.time()
    try:
        result = analyze_video(
            args.input,
            device=args.device,
            enable_mllm=args.enable_mllm,
            mllm_client=mllm_client,
        )

        print("\n" + "=" * 60)
        print("D5 物理常识分析结果")
        print("=" * 60)
        print(f"适用: {result['applicable']}")
        if not result["applicable"]:
            print(f"跳过原因: {result['skip_reason']}")
            return 0

        print(f"物理合理性评分: {result['physics_score']:.3f}")
        print(f"  - 漂移检测评分: {result['drift_score']:.3f}")
        if result["drift_events"]:
            print(f"    检测到 {len(result['drift_events'])} 个漂移事件:")
            for evt in result["drift_events"]:
                print(f"      - {evt['description']}")

        if result["vlm_score"] is not None:
            print(f"  - VLM 评分: {result['vlm_score']:.3f}")
            if result["vlm_reasoning"]:
                print(f"    推理过程: {result['vlm_reasoning']}")
            if result["vlm_violations"]:
                print(f"    检测到 {len(result['vlm_violations'])} 个物理违规:")
                for vio in result["vlm_violations"]:
                    print(f"      - [{vio['type']}] {vio['description']} (严重度: {vio['severity']})")

            # 打印提示词 + 完整模型回复，并保存
            prompt = build_physics_prompt(drift_events=result.get("drift_events"))
            mllm_payload = {
                "prompt": prompt,
                "response": {
                    "vlm_score": result["vlm_score"],
                    "vlm_reasoning": result["vlm_reasoning"],
                    "vlm_violations": result["vlm_violations"],
                },
            }
            stem = Path(args.input).stem
            mllm_out = Path(args.save_json).parent if args.save_json else REPO_ROOT / "outputs" / "physics"
            mllm_out.mkdir(parents=True, exist_ok=True)
            mllm_path = mllm_out / f"{stem}_mllm_prompt_response.json"
            mllm_path.write_text(json.dumps(mllm_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n{'='*60}")
            print("[MLLM 调用: 物理常识判定]")
            print(f"{'='*60}")
            print("[提示词]:")
            print(prompt)
            print(f"\n[模型完整回复]:")
            print(json.dumps(mllm_payload["response"], ensure_ascii=False, indent=2))
            print(f"\n提示词+回复已保存: {mllm_path}")
        else:
            print("  - VLM 评分: 未启用或不可用")

        elapsed = time.time() - t_total
        result["elapsed_sec"] = round(elapsed, 3)
        print(f"\n总耗时: {elapsed:.1f}s")

        if args.save_json:
            Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save_json).write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"结果已保存: {args.save_json}")

        return 0

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
