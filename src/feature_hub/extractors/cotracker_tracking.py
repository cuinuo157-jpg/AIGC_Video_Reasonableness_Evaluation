"""CoTracker 点追踪提取器（FeatureHub）。

默认尝试使用 CoTracker3 offline 权重进行网格点追踪；
若依赖/权重不可用，自动回退到 MediaPipe 关键点轨迹提取器。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .keypoint_tracking import extract_keypoint_trajectories

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_COTRACKER_DIR = _PROJECT_ROOT / "third_party" / "co-tracker"
_CHECKPOINT_CANDIDATES = [
    _PROJECT_ROOT / ".cache" / "scaled_offline.pth",
    _PROJECT_ROOT / ".cache" / "scaled_online.pth",
]


def _resolve_checkpoint() -> Path | None:
    for checkpoint in _CHECKPOINT_CANDIDATES:
        if checkpoint.exists():
            return checkpoint
    return None


def _load_frames(video_path: str, hub: Any = None) -> list[np.ndarray]:
    if hub is not None:
        try:
            frames = hub.get("video_frames")
            if frames:
                return frames
        except (KeyError, RuntimeError):
            pass

    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def _build_video_tensor(frames_bgr: list[np.ndarray], torch_mod: Any) -> tuple[Any, int, int]:
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    video_np = np.stack(rgb, axis=0)  # (T, H, W, 3)
    h, w = video_np.shape[1], video_np.shape[2]
    video_t = torch_mod.from_numpy(video_np).permute(0, 3, 1, 2)[None].float()  # (1, T, 3, H, W)
    return video_t, w, h


def _build_subject_mask_tensor(hub: Any, h: int, w: int, torch_mod: Any, device: str) -> Any | None:
    try:
        subject_result = hub.get("subject_masks")
        masks = getattr(subject_result, "masks", None)
        if not masks:
            return None

        first_valid: np.ndarray | None = None
        for mask in masks:
            if mask is None:
                continue
            m = np.asarray(mask).astype(bool)
            if m.any():
                first_valid = m
                break
        if first_valid is None:
            return None

        if first_valid.shape != (h, w):
            first_valid = cv2.resize(first_valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

        # CoTracker expects mask with channel dim: (1, 1, H, W).
        segm_mask = (first_valid.astype(np.uint8) * 255)[None, None, ...]
        return torch_mod.from_numpy(segm_mask).to(device)
    except (KeyError, RuntimeError, Exception):
        return None


def _to_trajectories(
    pred_tracks: Any,
    pred_visibility: Any,
    width: int,
    height: int,
) -> list[np.ndarray]:
    tracks = pred_tracks[0].detach().cpu().numpy()  # (T, N, 2)
    visibility = pred_visibility[0].detach().cpu().numpy().astype(bool)  # (T, N)

    t, n = tracks.shape[0], tracks.shape[1]
    trajectories: list[np.ndarray] = []
    for point_idx in range(n):
        traj = np.full((t, 2), np.nan, dtype=np.float32)
        visible = visibility[:, point_idx]
        if not np.any(visible):
            continue
        traj[visible, 0] = tracks[visible, point_idx, 0] / max(width - 1, 1)
        traj[visible, 1] = tracks[visible, point_idx, 1] / max(height - 1, 1)
        trajectories.append(traj)

    return trajectories


def extract_cotracker_trajectories(
    video_path: str,
    device: str,
    hub: Any = None,
) -> list[np.ndarray]:
    """提取 CoTracker 时序轨迹。

    Returns:
        list[np.ndarray]: 每条轨迹形状为 (T, 2)，不可见点为 NaN，坐标归一化到 [0, 1]。
    """
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch 不可用，tracking 回退到 keypoint_tracking")
        return extract_keypoint_trajectories(video_path, device, hub)

    if str(_COTRACKER_DIR) not in sys.path:
        sys.path.insert(0, str(_COTRACKER_DIR))

    try:
        from cotracker.predictor import CoTrackerPredictor
    except Exception as e:
        logger.warning("CoTracker 依赖不可用，tracking 回退到 keypoint_tracking: %s", e)
        return extract_keypoint_trajectories(video_path, device, hub)

    checkpoint = _resolve_checkpoint()
    if checkpoint is None:
        logger.warning(
            "CoTracker 权重未找到，tracking 回退到 keypoint_tracking。建议放置到: %s",
            " | ".join(str(p) for p in _CHECKPOINT_CANDIDATES),
        )
        return extract_keypoint_trajectories(video_path, device, hub)

    frames = _load_frames(video_path, hub)
    if len(frames) < 2:
        return []

    runtime_device = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    try:
        model = CoTrackerPredictor(
            checkpoint=str(checkpoint),
            v2=False,
            offline=True,
            window_len=60,
        ).to(runtime_device)

        video_t, width, height = _build_video_tensor(frames, torch)
        video_t = video_t.to(runtime_device)

        kwargs: dict[str, Any] = {
            "grid_size": 20,
            "grid_query_frame": 0,
            "backward_tracking": True,
        }
        if hub is not None:
            segm_mask = _build_subject_mask_tensor(hub, height, width, torch, runtime_device)
            if segm_mask is not None:
                kwargs["segm_mask"] = segm_mask

        with torch.no_grad():
            pred_tracks, pred_visibility = model(video_t, **kwargs)

        trajectories = _to_trajectories(pred_tracks, pred_visibility, width, height)
        if trajectories:
            return trajectories
    except Exception as e:
        logger.warning("CoTracker 推理失败，tracking 回退到 keypoint_tracking: %s", e)

    return extract_keypoint_trajectories(video_path, device, hub)
