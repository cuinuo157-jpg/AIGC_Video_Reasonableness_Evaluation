"""
时序连贯性 (TCS-lite) 调试脚本。

--enable-mllm 时默认 huawei_custom 图片帧接口；
也可通过 --mllm-provider 切换到 vllm / dashscope / openai / anthropic。

环境变量: MLLM_API_BASE_URL、MLLM_API_KEY、MLLM_API_SERVICE_NAME；
也兼容 VLLM_OPENAI_BASE_URL / VLLM_API_KEY 和 DASHSCOPE_API_KEY。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.feature_hub.hub import create_default_hub
from src.feature_hub.extractors.subject_segmentation import (
    _try_load_grounding_dino,
    _detect_boxes_grounding_dino,
)
from src.temporal_coherence.analyzer import TemporalCoherenceAnalyzer
from src.mllm.config import MLLMConfig
from src.mllm.client import MLLMClient
from src.mllm.dotenv_loader import load_dotenv

DEFAULT_MLLM_PROVIDER = "huawei_custom"
DEFAULT_MLLM_MODEL = "Qwen3-VL-32B-Instruct"
DEFAULT_MLLM_BASE_URL = "http://aitest-beta.rnd.huawei.com/v1"
DEFAULT_MLLM_SERVICE_NAME = "simple_client"


TEMPORAL_ANOMALY_CONFIRM_PROMPT = """你是一个视频时序一致性分析专家。视频中检测到以下疑似异常事件，请逐一判断是否为真实的 AI 生成异常。

## 疑似异常事件
{events_desc}

## 判断标准
- 物体凭空出现（非从画面边缘进入、非逐渐放大）→ 异常
- 物体突然消失（非从画面边缘离开、非逐渐缩小）→ 异常
- 物体出现/消失符合场景逻辑（如灯光开关、遮挡）→ 正常

请以 JSON 格式输出：
{{
    "has_anomalies": true或false,
    "anomaly_score": 0.0到1.0（0=完全正常，1=严重异常）,
    "judgements": [
        {{
            "event_index": 0,
            "is_anomaly": true或false,
            "reason": "判断理由"
        }}
    ]
}}"""

TEMPORAL_ANOMALY_DIRECT_PROMPT = """你是一个视频时序一致性分析专家。请直接分析这段视频，判断是否存在物体异常出现或消失的现象。

## 检查要点
- 物体是否凭空出现（非从画面边缘进入、非逐渐放大出现）
- 物体是否突然消失（非从画面边缘离开、非逐渐缩小消失）
- 背景元素（月亮、星星、建筑、树木等）是否突然出现或消失
- 前景物体（人、动物、车辆等）是否有不合理的出现/消失

请以 JSON 格式输出：
{{
    "has_anomalies": true或false,
    "anomaly_score": 0.0到1.0（0=完全正常，1=严重异常）,
    "anomalies": [
        {{
            "type": "appear或disappear",
            "object": "物体描述",
            "frame_range": "大约在哪个时间段",
            "reason": "为何判断为异常",
            "severity": "mild或moderate或severe"
        }}
    ]
}}"""



def _preview_and_save_mllm_frames(
    video_path: str,
    sample_fps: int,
    max_frames: int,
    save_dir: Path,
    label: str = "mllm",
) -> dict:
    """抽取将发送给 MLLM 的帧，打印抽帧统计，并将帧保存到磁盘。"""
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


def _call_vlm(hub, mllm_client: MLLMClient, prompt: str, video_path: Path) -> dict:
    """统一 VLM 调用；dashscope / vllm 优先走 judge_video_path（与物理模块一致）。"""
    provider = getattr(getattr(mllm_client, "config", None), "api_provider", "")
    if provider in ("dashscope", "vllm") and hasattr(mllm_client, "judge_video_path"):
        try:
            return mllm_client.judge_video_path(str(video_path), prompt)
        except Exception:
            pass
    try:
        frames = hub.get("video_frames")
        return mllm_client.judge_video_clip(frames, prompt)
    except Exception as e:
        return {"skipped": True, "reason": str(e)}


def _judge_anomalies_confirm(
    hub, abnormal_events: list, mllm_client: MLLMClient, video_path: Path
) -> dict:
    """confirm 模式：DINO 初筛后，VLM 对 abnormal 事件逐一确认。"""
    if not abnormal_events:
        return {"skipped": True, "reason": "no abnormal events"}
    events_desc = "\n".join(
        f"{i}. 帧 {e.frame_idx}：物体（track_id={e.track_id}）"
        f"{'突然出现' if e.event_type == 'appear' else '突然消失'}"
        f"，位置 bbox={[round(v, 1) for v in e.bbox]}"
        for i, e in enumerate(abnormal_events)
    )
    prompt = TEMPORAL_ANOMALY_CONFIRM_PROMPT.format(events_desc=events_desc)
    return _call_vlm(hub, mllm_client, prompt, video_path)


def _judge_anomalies_direct(hub, mllm_client: MLLMClient, video_path: Path) -> dict:
    """direct 模式：跳过 DINO，直接让 VLM 判断视频中是否有物体异常出现/消失。"""
    return _call_vlm(hub, mllm_client, TEMPORAL_ANOMALY_DIRECT_PROMPT, video_path)


def _serialize_result(result, vlm_result: dict | None = None) -> dict:
    out = {
        "applicable": result.applicable,
        "skip_reason": result.skip_reason,
        "temporal_coherence_score": result.temporal_coherence_score,
        "events_count": len(result.temporal_events),
        "abnormal_count": len(result.abnormal_events),
        "events": [
            {
                "event_type": e.event_type,
                "frame_idx": e.frame_idx,
                "track_id": e.track_id,
                "reason": e.reason,
                "bbox": e.bbox,
            }
            for e in result.temporal_events
        ],
    }
    if vlm_result:
        out["vlm_judgement"] = vlm_result
    return out


def _save_detection_visualizations(
    hub,
    analyzer: TemporalCoherenceAnalyzer,
    video_name: str,
    device: str,
    output_dir: Path,
) -> tuple[Path, int]:
    frames = hub.get("video_frames")
    vis_dir = output_dir / f"{video_name}_detections"
    vis_dir.mkdir(parents=True, exist_ok=True)
    gdino = _try_load_grounding_dino(device)
    if gdino is None:
        summary_path = vis_dir / f"{video_name}_detections.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump({"error": "grounding dino unavailable"}, f, ensure_ascii=False, indent=2)
        return vis_dir, 0
    gdino_model, gdino_transform = gdino

    n_saved = 0
    summary: dict[str, list[dict]] = {}
    sample_indices = list(range(0, len(frames), analyzer.config.sample_interval))
    if (len(frames) - 1) not in sample_indices:
        sample_indices.append(len(frames) - 1)
    for frame_idx in sorted(sample_indices):
        frame = frames[frame_idx].copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred = _detect_boxes_grounding_dino(
            frame_rgb,
            gdino_model,
            gdino_transform,
            device,
            return_semantics=True,
        )
        if pred is None:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels: list[str] = []
            scores = np.zeros((0,), dtype=np.float32)
        else:
            boxes, labels, scores = pred

        summary[str(frame_idx)] = []
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = labels[i] if i < len(labels) else "object"
            score = float(scores[i]) if i < len(scores) else 0.0
            summary[str(frame_idx)].append(
                {
                    "label": label,
                    "score": round(score, 4),
                    "bbox": [float(v) for v in b.tolist()],
                }
            )
            cv2.putText(
                frame,
                f"{label} {score:.2f}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame,
            f"frame={frame_idx} boxes={len(boxes)}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        out_path = vis_dir / f"{video_name}_det_{frame_idx:04d}.jpg"
        cv2.imwrite(str(out_path), frame)
        n_saved += 1
    summary_path = vis_dir / f"{video_name}_detections.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return vis_dir, n_saved


def run_one(
    video_path: Path,
    device: str,
    output_dir: Path,
    save_det_vis: bool = False,
    mllm_client: MLLMClient | None = None,
    vlm_mode: str = "confirm",  # "confirm" | "direct"
    mllm_fps: int = 2,
) -> Path:
    t0 = time.time()
    hub = create_default_hub(str(video_path), device=device)

    if mllm_client is not None:
        _preview_and_save_mllm_frames(
            video_path=str(video_path),
            sample_fps=mllm_fps,
            max_frames=5,
            save_dir=output_dir / f"{video_path.stem}_mllm_frames",
            label="D6 时序连贯性 VLM",
        )

    vlm_result = None

    if mllm_client and vlm_mode == "direct":
        # direct 模式：跳过 DINO，直接 VLM 判定
        print("  VLM direct 模式，跳过 Grounding DINO...")
        vlm_result = _judge_anomalies_direct(hub, mllm_client, video_path)
        result = TemporalCoherenceAnalyzer().analyze(hub)
    else:
        # confirm 模式（默认）：DINO 先跑，abnormal 事件再交 VLM 确认
        analyzer = TemporalCoherenceAnalyzer()
        result = analyzer.analyze(hub)
        if mllm_client and result.applicable and result.abnormal_events:
            print(f"  VLM confirm 模式，确认 {len(result.abnormal_events)} 个异常事件...")
            vlm_result = _judge_anomalies_confirm(hub, result.abnormal_events, mllm_client, video_path)

    elapsed = time.time() - t0

    payload = _serialize_result(result, vlm_result)
    payload["video_path"] = str(video_path)
    payload["elapsed_sec"] = round(elapsed, 3)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_path.stem}_tcs.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if save_det_vis:
        vis_dir, n_saved = _save_detection_visualizations(
            hub=hub,
            analyzer=TemporalCoherenceAnalyzer(),
            video_name=video_path.stem,
            device=device,
            output_dir=output_dir,
        )
        print(f"  det_vis_saved: {n_saved}")
        print(f"  det_vis_dir: {vis_dir}")

    print(f"[TCS] {video_path.name}")
    print(f"  applicable: {result.applicable}")
    if not result.applicable:
        print(f"  skip_reason: {result.skip_reason}")
    print(f"  score: {result.temporal_coherence_score:.4f}")
    print(f"  events: {len(result.temporal_events)}")
    print(f"  abnormal: {len(result.abnormal_events)}")
    if vlm_result and not vlm_result.get("skipped"):
        print(f"  vlm_anomaly_score: {vlm_result.get('anomaly_score', 'N/A')}")
        print(f"  vlm_has_anomalies: {vlm_result.get('has_anomalies', 'N/A')}")

        prompt = (
            TEMPORAL_ANOMALY_DIRECT_PROMPT
            if vlm_mode == "direct"
            else TEMPORAL_ANOMALY_CONFIRM_PROMPT.format(
                events_desc="\n".join(
                    f"{i}. 帧 {e.frame_idx}: track_id={e.track_id} {'出现' if e.event_type == 'appear' else '消失'}"
                    for i, e in enumerate(result.abnormal_events)
                )
            )
        )
        mllm_payload = {"prompt": prompt, "response": vlm_result}
        mllm_path = output_dir / f"{video_path.stem}_mllm_prompt_response.json"
        with mllm_path.open("w", encoding="utf-8") as f:
            json.dump(mllm_payload, f, ensure_ascii=False, indent=2)
        print(f"\n{'='*60}")
        print(f"[MLLM 调用: 时序连贯性 ({vlm_mode} 模式)]")
        print(f"{'='*60}")
        print("[提示词]:")
        print(prompt)
        print(f"\n[模型完整回复]:")
        print(json.dumps(vlm_result, ensure_ascii=False, indent=2))
        print(f"\n提示词+回复已保存: {mllm_path}")

    print(f"  saved: {out_path}")
    print(f"  elapsed: {elapsed:.1f}s")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="时序连贯性 (TCS-lite) 调试脚本")
    parser.add_argument("--input", required=True, help="视频文件路径")
    parser.add_argument("--device", default="cuda", help="推理设备 (cuda/cpu)")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "temporal_coherence"),
        help="结果输出目录",
    )
    parser.add_argument(
        "--save-det-vis",
        action="store_true",
        help="保存关键帧 GroundingDINO 检测框可视化",
    )
    parser.add_argument(
        "--enable-mllm",
        action="store_true",
        help=f"启用 VLM 判定（默认 {DEFAULT_MLLM_PROVIDER}）",
    )
    parser.add_argument(
        "--mllm-provider",
        default=os.environ.get("MLLM_PROVIDER", DEFAULT_MLLM_PROVIDER),
        choices=["vllm", "openai", "anthropic", "dashscope", "huawei_custom"],
        help=f"MLLM 提供方（默认 {DEFAULT_MLLM_PROVIDER}；可通过 MLLM_PROVIDER 环境变量配置）",
    )
    parser.add_argument(
        "--mllm-model",
        default=os.environ.get("MLLM_MODEL", DEFAULT_MLLM_MODEL),
        help=f"模型名（默认 {DEFAULT_MLLM_MODEL}；通过 MLLM_MODEL 环境变量配置）",
    )
    parser.add_argument(
        "--mllm-api-key",
        default=os.environ.get("MLLM_API_KEY", "")
        or os.environ.get("DASHSCOPE_API_KEY", "")
        or os.environ.get("VLLM_API_KEY", ""),
        help="API Key（openai/anthropic/dashscope 通常必填；vllm/huawei_custom 可空）",
    )
    parser.add_argument(
        "--mllm-base-url",
        default=os.environ.get("MLLM_API_BASE_URL", "")
        or os.environ.get("DASHSCOPE_BASE_URL", "")
        or os.environ.get("VLLM_OPENAI_BASE_URL", ""),
        help=f"Base URL（huawei_custom 默认 {DEFAULT_MLLM_BASE_URL}；vllm 默认代码内 localhost:8201/v1）",
    )
    parser.add_argument(
        "--mllm-service-name",
        default=os.environ.get("MLLM_API_SERVICE_NAME", DEFAULT_MLLM_SERVICE_NAME),
        help=f"自定义 API 的 service_name（默认 {DEFAULT_MLLM_SERVICE_NAME}；仅 huawei_custom 需要）",
    )
    parser.add_argument(
        "--mllm-fps",
        type=int,
        default=int(os.environ.get("MLLM_FPS", "2")),
        help="judge_video_path 抽帧 fps（通过 MLLM_FPS 环境变量配置）",
    )
    parser.add_argument(
        "--vlm-mode",
        choices=["confirm", "direct"],
        default="confirm",
        help=(
            "VLM 调用模式：\n"
            "  confirm = DINO 初筛后，仅对 reason=abnormal 的事件做 VLM 语义确认（省 API 费用）\n"
            "  direct  = 跳过 DINO，直接让 VLM 判断视频中是否有物体异常出现/消失"
        ),
    )
    args = parser.parse_args()

    load_dotenv()

    video_path = Path(args.input)
    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在: {video_path}")
    if video_path.is_dir():
        raise ValueError("当前脚本仅支持单视频输入，请传入具体视频文件")

    mllm_client = build_mllm_client(args)

    t_total = time.time()
    run_one(
        video_path,
        args.device,
        Path(args.output_dir),
        save_det_vis=args.save_det_vis,
        mllm_client=mllm_client,
        vlm_mode=args.vlm_mode,
        mllm_fps=args.mllm_fps,
    )
    print(f"\n总耗时: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
