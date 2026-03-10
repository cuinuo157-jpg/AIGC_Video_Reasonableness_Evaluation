from __future__ import annotations

import cv2
import numpy as np
import torch

_depth_model = None
_depth_transform = None


def _load_frames(video_path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def _get_depth_model(device: str) -> tuple:
    global _depth_model, _depth_transform
    if _depth_model is None:
        _depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        _depth_model.to(device).eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        _depth_transform = midas_transforms.dpt_transform
    return _depth_model, _depth_transform


def extract_depth_maps(video_path: str, device: str) -> list[np.ndarray]:
    frames = _load_frames(video_path)
    model, transform = _get_depth_model(device)
    depths = []
    for frame in frames:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(img_rgb).to(device)
        with torch.no_grad():
            prediction = model(input_batch)
        depth = prediction.squeeze().cpu().numpy()
        depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
        depths.append(depth)
    return depths
