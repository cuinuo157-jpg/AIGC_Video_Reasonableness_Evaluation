#!/usr/bin/env python3
"""
D5 物理常识与动力学一致性模块调试脚本。

使用 VLM（DashScope）进行物理常识判定。

依赖:
    pip install dashscope>=1.19.0
    或使用: uv sync

环境变量:
    DASHSCOPE_API_KEY   必填，百炼 API Key
    DASHSCOPE_BASE_URL  可选，国际区等需设置 endpoint

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


def build_mllm_client(
    api_key: str | None = None,
    api_provider: str = "dashscope",
    api_model: str = "qwen3-vl-8b-thinking",
) -> MLLMClient | None:
    """构建 MLLM 客户端。"""
    api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        print("警告: DASHSCOPE_API_KEY 未设置，将跳过 VLM 判定", file=sys.stderr)
        return None

    config = MLLMConfig(
        backend="api",
        api_provider=api_provider,
        api_model=api_model,
        api_key=api_key,
        api_base_url=os.environ.get("DASHSCOPE_BASE_URL", "").strip() or None,
    )
    return MLLMClient(config)


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
        help="启用 VLM 物理判定（需要 DASHSCOPE_API_KEY）",
    )
    p.add_argument(
        "--api-key",
        type=str,
        default="",
        help="DashScope API Key（默认读环境变量 DASHSCOPE_API_KEY）",
    )
    p.add_argument(
        "--api-model",
        type=str,
        default="qwen3-vl-8b-thinking",
        help="VLM 模型名",
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
        mllm_client = build_mllm_client(
            api_key=args.api_key or None,
            api_model=args.api_model,
        )
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
