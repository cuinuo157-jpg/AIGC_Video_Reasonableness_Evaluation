from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .config import TemporalCoherenceConfig
from src.feature_hub.extractors.subject_segmentation import (
    _try_load_grounding_dino,
    _detect_boxes_grounding_dino,
)


@dataclass
class TemporalEvent:
    event_type: str  # "appear" | "disappear"
    frame_idx: int
    track_id: int
    reason: str  # "edge_emerge" | "edge_vanish" | "small_emerge" | "small_vanish" | "detect_gap" | "abnormal"
    bbox: list[float]


@dataclass
class TemporalTrack:
    track_id: int
    frames: list[int] = field(default_factory=list)
    boxes: list[np.ndarray] = field(default_factory=list)


@dataclass
class TemporalCoherenceResult:
    applicable: bool
    skip_reason: str | None = None
    temporal_events: list[TemporalEvent] = field(default_factory=list)
    abnormal_events: list[TemporalEvent] = field(default_factory=list)
    temporal_coherence_score: float = 1.0


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(1e-6, float((a[2] - a[0]) * (a[3] - a[1])))
    area_b = max(1e-6, float((b[2] - b[0]) * (b[3] - b[1])))
    return float(inter / (area_a + area_b - inter + 1e-6))


def _bbox_area(b: np.ndarray) -> float:
    return float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))


def _touches_edge(b: np.ndarray, w: int, h: int, margin_ratio: float) -> bool:
    mx = w * margin_ratio
    my = h * margin_ratio
    return bool(
        b[0] <= mx or b[1] <= my or b[2] >= (w - mx) or b[3] >= (h - my)
    )


def _extract_boxes_with_gdino(
    frames_bgr: list[np.ndarray],
    device: str,
    config: TemporalCoherenceConfig,
) -> dict[int, list[np.ndarray]]:
    """抽样帧检测框。返回 {frame_idx: [xyxy, ...]}。"""
    gdino = _try_load_grounding_dino(device)
    if gdino is None:
        return {}
    model, transform = gdino

    detections: dict[int, list[np.ndarray]] = {}
    sample_indices = list(range(0, len(frames_bgr), config.sample_interval))
    if (len(frames_bgr) - 1) not in sample_indices:
        sample_indices.append(len(frames_bgr) - 1)

    for idx in sample_indices:
        frame = frames_bgr[idx]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = _detect_boxes_grounding_dino(frame_rgb, model, transform, device)
        if boxes is None or len(boxes) == 0:
            detections[idx] = []
            continue
        h, w = frame_rgb.shape[:2]
        min_area = config.min_box_area_ratio * (w * h)
        filtered = [b.astype(np.float32) for b in boxes if _bbox_area(b) >= min_area]
        detections[idx] = filtered
    return detections


def _track_boxes(
    detections: dict[int, list[np.ndarray]],
    config: TemporalCoherenceConfig,
) -> list[TemporalTrack]:
    """按 IoU 将抽样帧检测框串成轨迹。"""
    tracks: list[TemporalTrack] = []
    next_id = 0

    for frame_idx in sorted(detections.keys()):
        boxes = detections[frame_idx]
        used = [False] * len(boxes)

        # 先尝试匹配已有 track
        for tr in tracks:
            if not tr.frames:
                continue
            gap_steps = (frame_idx - tr.frames[-1]) // max(config.sample_interval, 1)
            if gap_steps > config.max_track_gap_steps + 1:
                continue

            best_j = -1
            best_iou = 0.0
            for j, b in enumerate(boxes):
                if used[j]:
                    continue
                iou = _bbox_iou(tr.boxes[-1], b)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= config.iou_match_threshold:
                tr.frames.append(frame_idx)
                tr.boxes.append(boxes[best_j])
                used[best_j] = True

        # 剩余框新建轨迹
        for j, b in enumerate(boxes):
            if used[j]:
                continue
            tracks.append(TemporalTrack(track_id=next_id, frames=[frame_idx], boxes=[b]))
            next_id += 1

    return tracks


def _classify_track_events(
    tracks: list[TemporalTrack],
    first_frame: int,
    last_frame: int,
    frame_w: int,
    frame_h: int,
    config: TemporalCoherenceConfig,
) -> list[TemporalEvent]:
    events: list[TemporalEvent] = []

    for tr in tracks:
        if not tr.frames:
            continue
        areas = [_bbox_area(b) for b in tr.boxes]
        start_frame, end_frame = tr.frames[0], tr.frames[-1]
        start_box, end_box = tr.boxes[0], tr.boxes[-1]

        # appear
        if start_frame > first_frame:
            if len(tr.frames) < config.min_track_len_steps:
                reason = "detect_gap"
            elif _touches_edge(start_box, frame_w, frame_h, config.edge_margin_ratio):
                reason = "edge_emerge"
            elif len(areas) >= 2 and (areas[1] / max(areas[0], 1e-6)) >= config.grow_ratio_threshold:
                reason = "small_emerge"
            else:
                reason = "abnormal"
            events.append(
                TemporalEvent(
                    event_type="appear",
                    frame_idx=start_frame,
                    track_id=tr.track_id,
                    reason=reason,
                    bbox=[float(x) for x in start_box.tolist()],
                )
            )

        # disappear
        if end_frame < last_frame:
            if len(tr.frames) < config.min_track_len_steps:
                reason = "detect_gap"
            elif _touches_edge(end_box, frame_w, frame_h, config.edge_margin_ratio):
                reason = "edge_vanish"
            elif len(areas) >= 2 and (areas[-1] / max(areas[-2], 1e-6)) <= config.shrink_ratio_threshold:
                reason = "small_vanish"
            else:
                reason = "abnormal"
            events.append(
                TemporalEvent(
                    event_type="disappear",
                    frame_idx=end_frame,
                    track_id=tr.track_id,
                    reason=reason,
                    bbox=[float(x) for x in end_box.tolist()],
                )
            )

    return events


class TemporalCoherenceAnalyzer:
    """TCS-lite: 检测目标异常出现/消失。"""

    def __init__(self, config: TemporalCoherenceConfig | None = None) -> None:
        self.config = config or TemporalCoherenceConfig()

    def analyze(self, hub: Any) -> TemporalCoherenceResult:
        frames = hub.get("video_frames")
        if not frames or len(frames) < 3:
            return TemporalCoherenceResult(
                applicable=False,
                skip_reason="insufficient frames",
            )

        device = getattr(hub, "device", "cpu")
        detections = _extract_boxes_with_gdino(frames, device, self.config)
        if not detections:
            return TemporalCoherenceResult(
                applicable=False,
                skip_reason="grounding dino unavailable",
            )

        tracks = _track_boxes(detections, self.config)
        h, w = frames[0].shape[:2]
        frame_keys = sorted(detections.keys())
        events = _classify_track_events(
            tracks,
            first_frame=frame_keys[0],
            last_frame=frame_keys[-1],
            frame_w=w,
            frame_h=h,
            config=self.config,
        )
        abnormal = [e for e in events if e.reason == "abnormal"]
        total_events = len(events)
        score = 1.0 if total_events == 0 else max(0.0, 1.0 - len(abnormal) / total_events)

        return TemporalCoherenceResult(
            applicable=True,
            temporal_events=events,
            abnormal_events=abnormal,
            temporal_coherence_score=float(np.clip(score, 0.0, 1.0)),
        )
