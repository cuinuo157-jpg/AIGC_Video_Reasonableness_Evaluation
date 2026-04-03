"""
动态度 (Dynamics) 单模块调试脚本

用法:
    python scripts/debug_dynamics.py --input <视频路径>
    python scripts/debug_dynamics.py --input <视频路径> --device cpu
    python scripts/debug_dynamics.py --input <视频路径> --method farneback
    python scripts/debug_dynamics.py --input <视频路径> --subject   # 启用主体分割
    python scripts/debug_dynamics.py --input data/videos/ --device cuda   # 批量

参数:
    --input        视频文件或目录路径
    --device       推理设备 (cuda / cpu)，默认 cuda
    --method       光流方法 (raft / farneback)，默认 raft
    --subject      启用主体分割 (SAM2/Grounding DINO)
    --save-vis     保存光流可视化到 outputs/dynamics/
    --max-frames   最大帧数（超出则均匀采样），默认 60
    --max-side     长边最大像素（RAFT 推荐 ≤ 512），默认 512
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

from src.motion_logic.dynamics_scorer import compute_dynamics_score, DynamicsDetail
from src.motion_logic.subject_motion_scorer import (
    compute_subject_motion_score,
    SubjectMotionDetail,
)
from src.motion_logic.trajectory_curvature_scorer import (
    TrajectoryCurvatureDetail,
    compute_trajectory_curvature_smoothness,
)


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


# ── 主体分割 ─────────────────────────────────────────────────

def extract_subject_masks_standalone(
    frames_rgb: list[np.ndarray], device: str, offline: bool = False
) -> tuple[list[np.ndarray], list[float], str]:
    """独立的主体分割，不依赖 FeatureHub。"""
    from src.feature_hub.extractors.subject_segmentation import (
        _try_load_sam2,
        _try_load_grounding_dino,
        _detect_boxes_grounding_dino,
        _segment_with_sam2_boxes,
        _segment_with_sam2_auto,
        _interpolate_masks,
        _SEGMENT_INTERVAL,
    )

    n = len(frames_rgb)
    masks: list[np.ndarray] = [np.zeros(frames_rgb[0].shape[:2], dtype=bool)] * n
    method = "none"

    sam2_predictor = _try_load_sam2(device)
    gdino = (
        _try_load_grounding_dino(device, offline=offline)
        if sam2_predictor is not None else None
    )

    if sam2_predictor is not None:
        sample_indices = list(range(0, n, _SEGMENT_INTERVAL))
        if (n - 1) not in sample_indices:
            sample_indices.append(n - 1)

        for idx in sample_indices:
            frame = frames_rgb[idx]
            mask = None

            if gdino is not None:
                processor, model = gdino
                boxes = _detect_boxes_grounding_dino(frame, processor, model, device)
                if boxes is not None and len(boxes) > 0:
                    mask = _segment_with_sam2_boxes(frame, sam2_predictor, boxes)
                    if method == "none":
                        method = "sam2_grounding"

            if mask is None or not mask.any():
                mask = _segment_with_sam2_auto(frame, sam2_predictor)
                if mask.any() and method == "none":
                    method = "sam2_auto"

            if mask is not None:
                masks[idx] = mask

        _interpolate_masks(masks, sample_indices)
    else:
        print("  [INFO] SAM2 不可用，主体分割跳过")

    ratios = [float(np.sum(m)) / (m.shape[0] * m.shape[1]) for m in masks]
    return masks, ratios, method


def extract_tracking_trajectories_standalone(
    video_path: str,
    frames_rgb: list[np.ndarray],
    device: str,
    masks: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """独立提取时序轨迹（优先 CoTracker，失败自动回退关键点轨迹）。"""
    from src.feature_hub.extractors.cotracker_tracking import (
        extract_cotracker_trajectories,
    )
    from src.feature_hub.extractors.subject_segmentation import SubjectSegmentationResult

    class _HubProxy:
        def __init__(
            self,
            frames_bgr: list[np.ndarray],
            subject_masks: SubjectSegmentationResult | None,
            keypoints_loader,
        ) -> None:
            self._frames_bgr = frames_bgr
            self._subject_masks = subject_masks
            self._keypoints_seq: list[dict] | None = None
            self._keypoints_loader = keypoints_loader

        def get(self, key: str):
            if key == "video_frames":
                return self._frames_bgr
            if key == "subject_masks" and self._subject_masks is not None:
                return self._subject_masks
            if key == "keypoints":
                if self._keypoints_seq is None:
                    try:
                        self._keypoints_seq = self._keypoints_loader()
                    except Exception as e:  # noqa: BLE001
                        # Keep debug pipeline robust even if MediaPipe/protobuf
                        # environment is incompatible on this machine.
                        print(f"  [WARN] keypoints 提取失败，轨迹回退不可用: {e}")
                        self._keypoints_seq = []
                return self._keypoints_seq
            raise KeyError(key)

    frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames_rgb]
    subject_result = None
    if masks is not None:
        subject_result = SubjectSegmentationResult(
            masks=masks,
            subject_ratios=[],
            method="external",
        )

    # CoTracker failure path may request hub.get("keypoints"). We use lazy loading
    # so MediaPipe issues won't break the whole dynamics debug run.
    def _load_keypoints():
        from src.feature_hub.extractors.mediapipe_keypoints import (
            extract_mediapipe_keypoints,
        )

        return extract_mediapipe_keypoints(video_path, device)

    hub_proxy = _HubProxy(frames_bgr, subject_result, _load_keypoints)
    return extract_cotracker_trajectories(video_path, device, hub_proxy)


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
    masks: list[np.ndarray] | None = None,
    trajectories: list[np.ndarray] | None = None,
    trajectory_detail: TrajectoryCurvatureDetail | None = None,
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

        # 主体 mask 叠加 + mask 内光流
        if masks is not None and i < len(masks) and masks[i].any():
            mask = masks[i]
            if mask.shape != vis.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8), (vis.shape[1], vis.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            # mask 叠加图: 主体区域半透明绿色
            overlay = vis.copy()
            overlay[mask] = (overlay[mask] * 0.5 + np.array([0, 128, 0]) * 0.5).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"{video_name}_subject_{i:04d}.png"), overlay)

            # mask 内光流彩色图
            masked_flow = flow_to_color(
                flows[i][0] * mask.astype(float),
                flows[i][1] * mask.astype(float),
            )
            cv2.imwrite(str(out_dir / f"{video_name}_subject_flow_{i:04d}.png"), masked_flow)

    # 保存平均光流幅度热力图
    mean_mag = np.mean(
        [np.sqrt(fx ** 2 + fy ** 2) for fx, fy in flows], axis=0
    )
    heatmap = cv2.applyColorMap(
        np.clip(mean_mag * 10, 0, 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    cv2.imwrite(str(out_dir / f"{video_name}_mean_mag.png"), heatmap)

    if trajectories:
        _save_trajectory_curvature_vis(
            trajectories,
            out_dir / f"{video_name}_trajectory_curvature.png",
            out_dir / f"{video_name}_trajectory_curvature.json",
            trajectory_detail=trajectory_detail,
            image_shape=flows[0][0].shape,
        )
    print(f"  可视化已保存到 {out_dir}")


def _save_trajectory_curvature_vis(
    trajectories: list[np.ndarray],
    image_path: Path,
    json_path: Path,
    trajectory_detail: TrajectoryCurvatureDetail | None,
    image_shape: tuple[int, int],
) -> None:
    """保存轨迹曲率可视化（归一化轨迹渲染到画布）。"""
    h, w = image_shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)

    max_draw = min(len(trajectories), 100)
    for idx in range(max_draw):
        traj = trajectories[idx]
        if traj.ndim != 2 or traj.shape[1] != 2:
            continue
        valid = np.all(np.isfinite(traj), axis=1)
        pts = traj[valid]
        if len(pts) < 2:
            continue
        px = np.clip((pts[:, 0] * (w - 1)).astype(np.int32), 0, w - 1)
        py = np.clip((pts[:, 1] * (h - 1)).astype(np.int32), 0, h - 1)
        poly = np.stack([px, py], axis=1).reshape(-1, 1, 2)
        color = (
            int((37 * idx) % 255),
            int((83 * idx + 80) % 255),
            int((127 * idx + 160) % 255),
        )
        cv2.polylines(canvas, [poly], isClosed=False, color=color, thickness=1)
        cv2.circle(canvas, tuple(poly[-1, 0]), 2, color, -1)

    lines = [
        f"tracks(total/valid): {len(trajectories)}/{getattr(trajectory_detail, 'valid_trajectory_count', 0)}",
        f"curvature_score: {getattr(trajectory_detail, 'score', 1.0):.4f}",
        f"abnormal_events: {getattr(trajectory_detail, 'abnormal_event_count', 0)}",
        f"abnormal_ratio: {getattr(trajectory_detail, 'abnormal_ratio', 0.0):.4f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (10, 25 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(image_path), canvas)
    payload = {
        "trajectory_count": len(trajectories),
        "visualized_track_count": max_draw,
        "trajectory_curvature_score": getattr(trajectory_detail, "score", 1.0),
        "valid_trajectory_count": getattr(trajectory_detail, "valid_trajectory_count", 0),
        "abnormal_event_count": getattr(trajectory_detail, "abnormal_event_count", 0),
        "abnormal_ratio": getattr(trajectory_detail, "abnormal_ratio", 0.0),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ── 主流程 ────────────────────────────────────────────────────

def analyze_video(
    video_path: str,
    device: str,
    method: str,
    save_vis: bool,
    enable_subject: bool = False,
    offline: bool = False,
    max_frames: int = 60,
    max_side: int = 512,
) -> DynamicsDetail:
    name = Path(video_path).stem
    print(f"\n{'='*60}")
    print(f"  视频: {Path(video_path).name}")
    print(f"  光流: {method.upper()}")
    if enable_subject:
        print(f"  主体分割: 启用")
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

    # 主体分割 + 可感知运动评分
    subject_detail: SubjectMotionDetail | None = None
    masks: list[np.ndarray] | None = None
    if enable_subject:
        t0 = time.time()
        masks, ratios, seg_method = extract_subject_masks_standalone(
            frames, device, offline=offline
        )
        t_seg = time.time() - t0
        print(f"  主体分割: {seg_method} ({t_seg:.1f}s)")

        if seg_method != "none":
            _, subject_detail = compute_subject_motion_score(flows, masks, ratios)

    t0 = time.time()
    trajectories = extract_tracking_trajectories_standalone(
        video_path, frames, device, masks=masks
    )
    t_track = time.time() - t0
    traj_score, traj_detail = compute_trajectory_curvature_smoothness(trajectories)
    print(f"  轨迹提取: {len(trajectories)} 条 ({t_track:.1f}s)")

    score, detail = compute_dynamics_score(
        flows, subject_motion=subject_detail
    )

    print(f"\n  ── 分量得分 ──")
    print(f"  光流幅度     (flow_magnitude):     {detail.flow_magnitude:.4f}")
    print(f"  空间覆盖率   (spatial_coverage):    {detail.spatial_coverage:.4f}")
    print(f"  时序变化     (temporal_variation):  {detail.temporal_variation:.4f}")
    print(f"  空间一致性   (spatial_consistency): {detail.spatial_consistency:.4f}")
    print(f"  相机因子     (camera_factor):       {detail.camera_factor:.4f}")
    if detail.subject_perceptual is not None:
        print(f"  主体可感知   (subject_perceptual): {detail.subject_perceptual:.4f}")
    print(f"  轨迹曲率平滑 (trajectory_curvature): {traj_score:.4f}")
    print(f"  场景类型:     {detail.scene_type}")
    print(f"\n  >>> 动态度总分: {detail.unified_score:.4f}")
    print(f"  >>> {detail.interpretation}")

    # 主体运动详情
    if subject_detail is not None:
        print(f"\n  ── 主体运动详情 ──")
        print(f"  主体运动幅度:   {subject_detail.subject_magnitude:.3f} px/frame")
        print(f"  背景运动幅度:   {subject_detail.background_magnitude:.3f} px/frame")
        print(f"  可感知得分:     {subject_detail.perceptual_score:.4f}")
        print(f"  主体平均占比:   {subject_detail.subject_ratio_mean:.2%}")

    if traj_detail.trajectory_count > 0:
        print(f"\n  ── 轨迹曲率详情 ──")
        print(f"  总轨迹数:       {traj_detail.trajectory_count}")
        print(f"  有效轨迹数:     {traj_detail.valid_trajectory_count}")
        print(f"  异常事件数:     {traj_detail.abnormal_event_count}")
        print(f"  异常事件占比:   {traj_detail.abnormal_ratio:.4f}")

    # 额外统计
    mags = [float(np.mean(np.sqrt(fx**2 + fy**2))) for fx, fy in flows]
    print(f"\n  ── 光流统计 ──")
    print(f"  均值: {np.mean(mags):.3f} px/frame")
    print(f"  标准差: {np.std(mags):.3f}")
    print(f"  最大: {np.max(mags):.3f}")
    print(f"  最小: {np.min(mags):.3f}")

    if save_vis:
        out_dir = ROOT / "outputs" / "dynamics"
        save_visualizations(
            flows,
            detail,
            name,
            out_dir,
            masks=masks,
            trajectories=trajectories,
            trajectory_detail=traj_detail,
        )

    return detail


def main() -> None:
    parser = argparse.ArgumentParser(description="动态度检测调试脚本")
    parser.add_argument("--input", required=True, help="视频文件或目录")
    parser.add_argument("--device", default="cuda", help="推理设备")
    parser.add_argument("--method", default="raft", choices=["raft", "farneback"])
    parser.add_argument("--subject", action="store_true", help="启用主体分割 (SAM2)")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="离线模式：禁止联网下载模型资源（缺本地资源时会失败）",
    )
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
    if args.subject:
        print("主体分割: 启用 (SAM2 + Grounding DINO)")
    if args.offline:
        os.environ["AIGC_OFFLINE_MODE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("离线模式: 启用（仅使用本地 .cache 资源）")

    results: list[tuple[str, DynamicsDetail]] = []
    for v in videos:
        detail = analyze_video(
            str(v), args.device, args.method, args.save_vis,
            enable_subject=args.subject,
            offline=args.offline,
            max_frames=args.max_frames, max_side=args.max_side,
        )
        results.append((v.name, detail))

    if len(results) > 1:
        print(f"\n{'='*60}")
        print("  汇总")
        print(f"{'='*60}")
        header = f"  {'视频':<45} {'动态度':>6} {'场景':>8}"
        if args.subject:
            header += f" {'主体感知':>8}"
        print(header)
        print(f"  {'-'*45} {'-'*6} {'-'*8}" + (" " + "-"*8 if args.subject else ""))
        for name, d in results:
            short = name[:42] + "..." if len(name) > 45 else name
            line = f"  {short:<45} {d.unified_score:>6.3f} {d.scene_type:>8}"
            if args.subject and d.subject_perceptual is not None:
                line += f" {d.subject_perceptual:>8.3f}"
            print(line)


if __name__ == "__main__":
    main()
