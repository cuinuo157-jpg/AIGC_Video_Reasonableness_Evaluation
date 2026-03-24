from __future__ import annotations

import argparse
import json
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
from src.temporal_coherence.analyzer import (
    TemporalCoherenceAnalyzer,
)


def _serialize_result(result) -> dict:
    return {
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
) -> Path:
    t0 = time.time()
    hub = create_default_hub(str(video_path), device=device)
    analyzer = TemporalCoherenceAnalyzer()
    result = analyzer.analyze(hub)
    elapsed = time.time() - t0

    payload = _serialize_result(result)
    payload["video_path"] = str(video_path)
    payload["elapsed_sec"] = round(elapsed, 3)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_path.stem}_tcs.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if save_det_vis:
        vis_dir, n_saved = _save_detection_visualizations(
            hub=hub,
            analyzer=analyzer,
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
    args = parser.parse_args()

    video_path = Path(args.input)
    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在: {video_path}")
    if video_path.is_dir():
        raise ValueError("当前脚本仅支持单视频输入，请传入具体视频文件")

    run_one(
        video_path,
        args.device,
        Path(args.output_dir),
        save_det_vis=args.save_det_vis,
    )


if __name__ == "__main__":
    main()
