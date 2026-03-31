"""
MediaPipe Iris 瞳孔追踪调试脚本（基于 FeatureHub）。

用法:
    python scripts/debug_iris_tracking.py --input <视频路径>
    python scripts/debug_iris_tracking.py --input <视频路径> --device cpu --save-vis

参数:
    --input        视频文件路径
    --device       推理设备 (cuda / cpu)，默认 cpu
    --save-vis     保存可视化结果到 outputs/iris_tracking/
    --vis-step     每 N 帧保存 1 张可视化图片，默认 10
    --save-video   保存整段可视化视频（需配合 --save-vis）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.feature_hub.hub import create_default_hub


def _to_pixel(pt: np.ndarray, w: int, h: int) -> tuple[int, int]:
    return int(np.clip(pt[0], 0.0, 1.0) * (w - 1)), int(np.clip(pt[1], 0.0, 1.0) * (h - 1))


def _draw_iris_overlay(frame: np.ndarray, iris: dict) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]
    scale = 0.5 * (w + h)

    left_center = iris.get("left_pupil_center")
    right_center = iris.get("right_pupil_center")
    left_radius = iris.get("left_pupil_radius")
    right_radius = iris.get("right_pupil_radius")
    ipd = iris.get("interpupil_distance")

    if left_center is not None:
        cx, cy = _to_pixel(left_center, w, h)
        cv2.circle(vis, (cx, cy), 3, (0, 255, 255), -1, cv2.LINE_AA)
        if left_radius is not None:
            cv2.circle(vis, (cx, cy), max(1, int(left_radius * scale)), (0, 200, 255), 1, cv2.LINE_AA)

    if right_center is not None:
        cx, cy = _to_pixel(right_center, w, h)
        cv2.circle(vis, (cx, cy), 3, (255, 255, 0), -1, cv2.LINE_AA)
        if right_radius is not None:
            cv2.circle(vis, (cx, cy), max(1, int(right_radius * scale)), (255, 200, 0), 1, cv2.LINE_AA)

    if left_center is not None and right_center is not None:
        lxy = _to_pixel(left_center, w, h)
        rxy = _to_pixel(right_center, w, h)
        cv2.line(vis, lxy, rxy, (0, 255, 0), 1, cv2.LINE_AA)

    text = []
    text.append(f"IPD={ipd:.4f}" if ipd is not None else "IPD=N/A")
    lrn = iris.get("left_pupil_radius_norm")
    rrn = iris.get("right_pupil_radius_norm")
    text.append(f"Lr={lrn:.4f}" if lrn is not None else "Lr=N/A")
    text.append(f"Rr={rrn:.4f}" if rrn is not None else "Rr=N/A")
    cv2.putText(
        vis,
        "  ".join(text),
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return vis


def _safe_mean(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def main() -> None:
    parser = argparse.ArgumentParser(description="MediaPipe Iris 瞳孔追踪调试")
    parser.add_argument("--input", required=True, help="视频文件路径")
    parser.add_argument("--device", default="cpu", help="推理设备 (cuda/cpu)")
    parser.add_argument("--save-vis", action="store_true", help="保存可视化结果")
    parser.add_argument("--vis-step", type=int, default=10, help="每 N 帧保存 1 张可视化图片")
    parser.add_argument("--save-video", action="store_true", help="保存整段可视化视频")
    args = parser.parse_args()

    video_path = Path(args.input)
    if not video_path.exists():
        print(f"错误: 视频文件不存在 {video_path}")
        sys.exit(1)

    print("=" * 60)
    print("Iris 瞳孔追踪调试")
    print("=" * 60)
    print(f"输入视频: {video_path}")
    print(f"设备: {args.device}")

    hub = create_default_hub(str(video_path), device=args.device)
    frames = hub.get("video_frames")
    iris_seq = hub.get("iris_tracking")
    n = min(len(frames), len(iris_seq))
    if n == 0:
        print("错误: 无有效帧或无 iris 结果")
        sys.exit(1)

    left_detected = sum(1 for x in iris_seq[:n] if x.get("left_pupil_center") is not None)
    right_detected = sum(1 for x in iris_seq[:n] if x.get("right_pupil_center") is not None)
    both_detected = sum(
        1
        for x in iris_seq[:n]
        if x.get("left_pupil_center") is not None and x.get("right_pupil_center") is not None
    )
    avg_ipd = _safe_mean([x.get("interpupil_distance") for x in iris_seq[:n]])
    avg_lrn = _safe_mean([x.get("left_pupil_radius_norm") for x in iris_seq[:n]])
    avg_rrn = _safe_mean([x.get("right_pupil_radius_norm") for x in iris_seq[:n]])

    print(f"\n总帧数: {n}")
    print(f"左眼检出: {left_detected}/{n} ({left_detected / n:.1%})")
    print(f"右眼检出: {right_detected}/{n} ({right_detected / n:.1%})")
    print(f"双眼检出: {both_detected}/{n} ({both_detected / n:.1%})")
    print(f"平均瞳距 IPD: {avg_ipd:.5f}" if avg_ipd is not None else "平均瞳距 IPD: N/A")
    print(f"平均左眼瞳孔半径(归一化): {avg_lrn:.5f}" if avg_lrn is not None else "平均左眼瞳孔半径(归一化): N/A")
    print(f"平均右眼瞳孔半径(归一化): {avg_rrn:.5f}" if avg_rrn is not None else "平均右眼瞳孔半径(归一化): N/A")

    if not args.save_vis:
        print("\n完成（仅统计）。如需可视化请加 --save-vis。")
        return

    out_dir = ROOT / "outputs" / "iris_tracking" / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存关键帧可视化
    vis_step = max(1, args.vis_step)
    for idx in range(0, n, vis_step):
        vis = _draw_iris_overlay(frames[idx], iris_seq[idx])
        cv2.imwrite(str(out_dir / f"frame_{idx:04d}.jpg"), vis)

    # 保存可视化视频
    if args.save_video:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(out_dir / "iris_overlay.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        for idx in range(n):
            writer.write(_draw_iris_overlay(frames[idx], iris_seq[idx]))
        writer.release()

    print(f"\n可视化输出目录: {out_dir}")
    print("完成。")


if __name__ == "__main__":
    main()
