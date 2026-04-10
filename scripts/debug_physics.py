#!/usr/bin/env python3
"""
D5 物理常识与动力学一致性模块调试脚本。

启用 --enable-mllm 时默认走本地 OpenAI 兼容 VLLM（与 scripts/test_qwen_35_video.py 同接口）；
可选用 --mllm-provider dashscope 调用百炼。

依赖:
    VLLM: 无额外 Python 包（openai 客户端）
    DashScope: pip install dashscope>=1.19.0

环境变量:
    VLLM_OPENAI_BASE_URL  默认 http://localhost:8201/v1
    VLLM_API_KEY          可空（本地常用 not-needed）
    DASHSCOPE_API_KEY     dashscope 时必填
    DASHSCOPE_BASE_URL    可选

用法示例:
    python scripts/debug_physics.py --input data/sample.mp4
    python scripts/debug_physics.py --input data/sample.mp4 --enable-mllm --save-json outputs/physics_result.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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


def build_mllm_client(args: argparse.Namespace) -> MLLMClient | None:
    """构建 MLLM 客户端（与 debug_dynamics 参数语义一致）。"""
    if not args.enable_mllm:
        return None
    api_key = (args.mllm_api_key or "").strip()
    if args.mllm_provider != "vllm" and not api_key:
        print(
            "错误: 启用 --enable-mllm 且非 vllm 时必须提供 --mllm-api-key 或设置 DASHSCOPE_API_KEY",
            file=sys.stderr,
        )
        return None
    cfg = MLLMConfig(
        backend="api",
        api_provider=args.mllm_provider,
        api_model=args.mllm_model,
        api_key=api_key or None,
        api_base_url=(args.mllm_base_url or "").strip() or None,
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
        help="启用 VLM 物理判定（默认 vllm：OpenAI 兼容本地服务）",
    )
    p.add_argument(
        "--mllm-provider",
        type=str,
        default="vllm",
        choices=["vllm", "openai", "anthropic", "dashscope"],
        help="MLLM 提供方（默认 vllm）",
    )
    p.add_argument(
        "--mllm-model",
        type=str,
        default="qwen3.5:9b",
        help="模型名（vllm 默认 qwen3.5:9b；dashscope 可传 qwen3-vl-8b-thinking 等）",
    )
    p.add_argument(
        "--mllm-api-key",
        type=str,
        default=os.environ.get("DASHSCOPE_API_KEY", "")
        or os.environ.get("VLLM_API_KEY", ""),
        help="API Key（dashscope 必填；vllm 可空）",
    )
    p.add_argument(
        "--mllm-base-url",
        type=str,
        default=os.environ.get("DASHSCOPE_BASE_URL", "")
        or os.environ.get("VLLM_OPENAI_BASE_URL", ""),
        help="Base URL（vllm 默认代码内 localhost:8201/v1）",
    )
    p.add_argument(
        "--mllm-fps",
        type=int,
        default=2,
        help="judge_video_path 抽帧 fps（dashscope / vllm）",
    )
    p.add_argument(
        "--save-json",
        type=str,
        default="",
        help="可选：保存结果为 JSON 文件",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _load_repo_dotenv()
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
                print(f"    推理过程: {result['vlm_reasoning'][:200]}...")
            if result["vlm_violations"]:
                print(f"    检测到 {len(result['vlm_violations'])} 个物理违规:")
                for vio in result["vlm_violations"]:
                    print(f"      - [{vio['type']}] {vio['description']} (严重度: {vio['severity']})")
        else:
            print("  - VLM 评分: 未启用或不可用")

        if args.save_json:
            Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save_json).write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"\n结果已保存: {args.save_json}")

        return 0

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
