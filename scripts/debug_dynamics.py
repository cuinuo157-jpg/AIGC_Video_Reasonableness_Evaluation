"""
动态度 (Dynamics) 单模块调试脚本

用法:
    python scripts/debug_dynamics.py --input <视频路径>
    python scripts/debug_dynamics.py --input <视频路径> --device cpu
    python scripts/debug_dynamics.py --input <视频路径> --method farneback
    python scripts/debug_dynamics.py --input data/videos/ --device cuda   # 批量

参数:
    --input        视频文件或目录路径
    --device       推理设备 (cuda / cpu)，默认 cuda
    --method       光流方法 (raft / farneback)，默认 raft
    --save-vis     保存光流可视化到 outputs/dynamics/
    --max-frames   最大帧数（超出则均匀采样），默认 60
    --max-side     长边最大像素（RAFT 推荐 ≤ 512），默认 512
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.motion_logic.dynamics_scorer import compute_dynamics_score, DynamicsDetail


# ── 光流提取 ─────────────────────────────────────────────────

def _extract_flows_raft(
    frames_rgb: list[np.ndarray], device: str
) -> list[tuple[np.ndarray, np.ndarray]]:
    from src.feature_hub.extractors.raft_flow import SimpleRAFT

    predictor = SimpleRAFT(device=device, method="raft")
    flows = []
    for i in range(len(frames_rgb) - 1):
        f = predictor.predict_flow(frames_rgb[i], frames_rgb[i + 1])  # (2,H,W)
        flows.append((f[0], f[1]))
    return flows


def _extract_flows_farneback(
    frames_rgb: list[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    flows = []
    for i in range(len(frames_rgb) - 1):
        g1 = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(frames_rgb[i + 1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append((flow[..., 0], flow[..., 1]))
    return flows


def extract_flows(
    frames_rgb: list[np.ndarray], device: str, method: str
) -> list[tuple[np.ndarray, np.ndarray]]:
    if method == "raft":
        return _extract_flows_raft(frames_rgb, device)
    return _extract_flows_farneback(frames_rgb)


# ── 视频加载 ─────────────────────────────────────────────────

def load_video_rgb(
    path: str, max_frames: int = 60, max_side: int = 512
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames) if total > max_frames else 1
    frames: list[np.ndarray] = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 缩放: 保持长边 ≤ max_side
            h, w = frame_rgb.shape[:2]
            if max(h, w) > max_side:
                scale = max_side / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
            frames.append(frame_rgb)
        idx += 1
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames


# ── 光流可视化 ────────────────────────────────────────────────

def flow_to_color(flow_x: np.ndarray, flow_y: np.ndarray) -> np.ndarray:
    """HSV 编码光流为彩色图。"""
    mag = np.sqrt(flow_x ** 2 + flow_y ** 2)
    ang = np.arctan2(flow_y, flow_x)
    hsv = np.zeros((*flow_x.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag * 8, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def save_visualizations(
    flows: list[tuple[np.ndarray, np.ndarray]],
    detail: DynamicsDetail,
    video_name: str,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # 保存前 5 帧和最后 1 帧的光流可视化
    indices = list(range(min(5, len(flows)))) + (
        [len(flows) - 1] if len(flows) > 5 else []
    )
    for i in set(indices):
        vis = flow_to_color(flows[i][0], flows[i][1])
        fname = out_dir / f"{video_name}_flow_{i:04d}.png"
        cv2.imwrite(str(fname), vis)

    # 保存平均光流幅度热力图
    mean_mag = np.mean(
        [np.sqrt(fx ** 2 + fy ** 2) for fx, fy in flows], axis=0
    )
    heatmap = cv2.applyColorMap(
        np.clip(mean_mag * 10, 0, 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    cv2.imwrite(str(out_dir / f"{video_name}_mean_mag.png"), heatmap)
    print(f"  可视化已保存到 {out_dir}")


# ── 主流程 ────────────────────────────────────────────────────

def analyze_video(
    video_path: str,
    device: str,
    method: str,
    save_vis: bool,
    max_frames: int = 60,
    max_side: int = 512,
) -> DynamicsDetail:
    name = Path(video_path).stem
    print(f"\n{'='*60}")
    print(f"  视频: {Path(video_path).name}")
    print(f"  光流: {method.upper()}")
    print(f"{'='*60}")

    t0 = time.time()
    frames = load_video_rgb(video_path, max_frames=max_frames, max_side=max_side)
    t_load = time.time() - t0
    if frames:
        print(f"  帧数: {len(frames)}, 分辨率: {frames[0].shape[1]}x{frames[0].shape[0]} ({t_load:.1f}s)")
    else:
        print(f"  帧数: 0 ({t_load:.1f}s)")

    if len(frames) < 2:
        print("  [SKIP] 帧数不足")
        return DynamicsDetail()

    t0 = time.time()
    flows = extract_flows(frames, device, method)
    t_flow = time.time() - t0
    print(f"  光流: {len(flows)} 帧 ({t_flow:.1f}s)")

    score, detail = compute_dynamics_score(flows)

    print(f"\n  ── 五分量得分 ──")
    print(f"  光流幅度     (flow_magnitude):     {detail.flow_magnitude:.4f}")
    print(f"  空间覆盖率   (spatial_coverage):    {detail.spatial_coverage:.4f}")
    print(f"  时序变化     (temporal_variation):  {detail.temporal_variation:.4f}")
    print(f"  空间一致性   (spatial_consistency): {detail.spatial_consistency:.4f}")
    print(f"  相机因子     (camera_factor):       {detail.camera_factor:.4f}")
    print(f"  场景类型:     {detail.scene_type}")
    print(f"\n  >>> 动态度总分: {detail.unified_score:.4f}")
    print(f"  >>> {detail.interpretation}")

    # 额外统计
    mags = [float(np.mean(np.sqrt(fx**2 + fy**2))) for fx, fy in flows]
    print(f"\n  ── 光流统计 ──")
    print(f"  均值: {np.mean(mags):.3f} px/frame")
    print(f"  标准差: {np.std(mags):.3f}")
    print(f"  最大: {np.max(mags):.3f}")
    print(f"  最小: {np.min(mags):.3f}")

    if save_vis:
        out_dir = ROOT / "outputs" / "dynamics"
        save_visualizations(flows, detail, name, out_dir)

    return detail


def main() -> None:
    parser = argparse.ArgumentParser(description="动态度检测调试脚本")
    parser.add_argument("--input", required=True, help="视频文件或目录")
    parser.add_argument("--device", default="cuda", help="推理设备")
    parser.add_argument("--method", default="raft", choices=["raft", "farneback"])
    parser.add_argument("--save-vis", action="store_true", help="保存光流可视化")
    parser.add_argument("--max-frames", type=int, default=60, help="最大帧数")
    parser.add_argument("--max-side", type=int, default=512, help="长边最大像素")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        videos = sorted(
            p for p in input_path.rglob("*")
            if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        )
    else:
        videos = [input_path]

    if not videos:
        print("未找到视频文件")
        return

    print(f"共 {len(videos)} 个视频，设备: {args.device}，方法: {args.method}")

    results: list[tuple[str, DynamicsDetail]] = []
    for v in videos:
        detail = analyze_video(
            str(v), args.device, args.method, args.save_vis,
            max_frames=args.max_frames, max_side=args.max_side,
        )
        results.append((v.name, detail))

    if len(results) > 1:
        print(f"\n{'='*60}")
        print("  汇总")
        print(f"{'='*60}")
        print(f"  {'视频':<45} {'动态度':>6} {'场景':>8}")
        print(f"  {'-'*45} {'-'*6} {'-'*8}")
        for name, d in results:
            short = name[:42] + "..." if len(name) > 45 else name
            print(f"  {short:<45} {d.unified_score:>6.3f} {d.scene_type:>8}")


if __name__ == "__main__":
    main()
