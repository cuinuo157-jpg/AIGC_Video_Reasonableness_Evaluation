from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from enum import Enum

from src.temporal_reasoning.motion_flow.flow_analyzer import MotionFlowAnalyzer
from src.temporal_reasoning.core.config import RAFTConfig, KeypointConfig
from src.temporal_reasoning.keypoint_analysis.keypoint_analyzer import KeypointAnalyzer
from src.temporal_reasoning.region_analysis.region_temporal_change_detector import (
    RegionTemporalChangeConfig,
    RegionTemporalChangeDetector,
)


_COMBINED_COLORS = [
    (255, 99, 71),    # tomato
    (60, 179, 113),   # medium sea green
    (65, 105, 225),   # royal blue
    (238, 130, 238),  # violet
    (255, 215, 0),    # gold
    (70, 130, 180),   # steel blue
    (244, 164, 96),   # sandy brown
    (0, 206, 209),    # dark turquoise
    (205, 92, 92),    # indian red
    (123, 104, 238),  # medium slate blue
]


class CombinedMaskVisualizer:
    """Aggregate region masks and render combined overlays per frame."""

    def __init__(
        self,
        base_dir: Path,
        frames_uint8: Sequence[np.ndarray],
        max_frames: Optional[int],
    ) -> None:
        self._frames = frames_uint8
        self._max_frames = max_frames or 0
        self.output_dir = (base_dir / "combined").resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[int, Dict[str, object]] = {}
        self._order: List[int] = []

    def add_region_mask(self, frame_idx: int, region_name: str, mask: Optional[np.ndarray]) -> None:
        if mask is None:
            return
        if frame_idx < 0 or frame_idx >= len(self._frames):
            return

        if frame_idx not in self._entries:
            if self._max_frames > 0 and len(self._entries) >= self._max_frames:
                return
            self._entries[frame_idx] = {
                "frame": self._frames[frame_idx].copy(),
                "regions": [],
            }
            self._order.append(frame_idx)

        regions: List[Tuple[str, np.ndarray]] = self._entries[frame_idx]["regions"]  # type: ignore[assignment]
        regions.append((region_name, mask.copy()))

    def save(self) -> Optional[Path]:
        if not self._entries:
            return None

        for frame_idx in self._order:
            entry = self._entries[frame_idx]
            frame_rgb: np.ndarray = entry["frame"]  # type: ignore[assignment]
            regions: List[Tuple[str, np.ndarray]] = entry["regions"]  # type: ignore[assignment]

            overlay = frame_rgb.copy()
            labels: List[Tuple[int, int, str]] = []

            for region_name, mask in regions:
                mask_bool = mask.astype(bool)
                if not np.any(mask_bool):
                    continue

                color = self._color_for_region(region_name)
                overlay[mask_bool] = color

                mask_uint8 = (mask_bool.astype(np.uint8) * 255)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 2)

                ys, xs = np.where(mask_bool)
                if xs.size > 0 and ys.size > 0:
                    labels.append((int(xs.mean()), int(ys.mean()), region_name))

            blended = cv2.addWeighted(frame_rgb, 0.6, overlay, 0.4, 0)

            for x, y, name in labels:
                cv2.putText(
                    blended,
                    name,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            save_path = self.output_dir / f"frame_{frame_idx:04d}.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

        return self.output_dir

    @staticmethod
    def _color_for_region(region_name: str) -> Tuple[int, int, int]:
        idx = abs(hash(region_name)) % len(_COMBINED_COLORS)
        return _COMBINED_COLORS[idx]



class RegionMaskMode(Enum):
    POLYGON = "polygon"
    CONVEX_HULL = "convex_hull"
    BOUNDING_BOX = "bounding_box"


@dataclass
class RegionDefinition:
    name: str
    keypoint_group: str
    mask_mode: RegionMaskMode
    keypoint_indices: Sequence[int] = ()
    min_area: int = 64
    margin_ratio: float = 0.02
    temporal_config: RegionTemporalChangeConfig = field(default_factory=RegionTemporalChangeConfig)


def default_regions() -> List[RegionDefinition]:
    return [
        RegionDefinition(
            name="mouth",
            keypoint_group="face",
            mask_mode=RegionMaskMode.POLYGON,
            keypoint_indices=(61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308),
            min_area=64,
            margin_ratio=0.02,
            temporal_config=RegionTemporalChangeConfig(
                motion_threshold=2.0,
                similarity_threshold=0.25,
                hist_diff_threshold=0.012,
                consecutive_frames=1,
            ),
        ),
        RegionDefinition(
            name="left_eye",
            keypoint_group="face",
            mask_mode=RegionMaskMode.POLYGON,
            keypoint_indices=(33, 246, 161, 160, 159, 158, 157, 173),
            min_area=20,
            margin_ratio=0.02,
            temporal_config=RegionTemporalChangeConfig(
                motion_threshold=1.5,
                similarity_threshold=0.15,
                hist_diff_threshold=0.01,
                consecutive_frames=1,
            ),
        ),
        RegionDefinition(
            name="right_eye",
            keypoint_group="face",
            mask_mode=RegionMaskMode.POLYGON,
            keypoint_indices=(362, 398, 384, 385, 386, 387, 388, 466),
            min_area=20,
            margin_ratio=0.02,
            temporal_config=RegionTemporalChangeConfig(
                motion_threshold=1.5,
                similarity_threshold=0.15,
                hist_diff_threshold=0.01,
                consecutive_frames=1,
            ),
        ),
        RegionDefinition(
            name="left_hand",
            keypoint_group="left_hand",
            mask_mode=RegionMaskMode.CONVEX_HULL,
            min_area=50,
            margin_ratio=0.02,
            temporal_config=RegionTemporalChangeConfig(
                motion_threshold=2.5,
                similarity_threshold=0.3,
                hist_diff_threshold=0.015,
                consecutive_frames=1,
            ),
        ),
        RegionDefinition(
            name="right_hand",
            keypoint_group="right_hand",
            mask_mode=RegionMaskMode.CONVEX_HULL,
            min_area=50,
            margin_ratio=0.02,
            temporal_config=RegionTemporalChangeConfig(
                motion_threshold=2.5,
                similarity_threshold=0.3,
                hist_diff_threshold=0.015,
                consecutive_frames=1,
            ),
        ),
    ]


@dataclass
class RegionAnalysisPipelineConfig:
    raft: RAFTConfig
    keypoint: KeypointConfig = KeypointConfig()
    regions: List[RegionDefinition] = field(default_factory=default_regions)
    enable_visualization: bool = False
    visualization_output_dir: Optional[str] = None
    visualization_max_frames: Optional[int] = 150
    per_region_visualization: bool = False


class RegionAnalysisPipeline:
    """Generate region masks from MediaPipe keypoints and detect temporal anomalies."""

    def __init__(self, config: RegionAnalysisPipelineConfig):
        self.config = config
        self.flow_analyzer = MotionFlowAnalyzer(config.raft)
        self.keypoint_analyzer = KeypointAnalyzer(config.keypoint)
        self._initialized = False
        self._vis_dir: Optional[Path] = None
        self._vis_counters: Dict[str, int] = {}
        self._combined_visualizer: Optional[CombinedMaskVisualizer] = None

    def initialize(self) -> None:
        if self._initialized:
            return
        self.flow_analyzer.initialize()
        self.keypoint_analyzer.initialize()
        self._initialized = True

    def analyze(
        self,
        video_frames: Sequence[np.ndarray],
        fps: float = 30.0,
        video_path: Optional[str] = None,
        simple_output: bool = False,
    ) -> Dict[str, object]:
        if not self._initialized:
            self.initialize()
        self._prepare_visualization(video_path)
        if hasattr(self.keypoint_analyzer.extractor, "reset_timestamp"):
            self.keypoint_analyzer.extractor.reset_timestamp()

        frames_uint8 = [self._ensure_uint8(frame) for frame in video_frames]
        keypoint_sequence = [
            self.keypoint_analyzer.extractor.extract_keypoints(frame_uint8, fps=fps)
            for frame_uint8 in frames_uint8
        ]

        if self._vis_dir:
            self._combined_visualizer = CombinedMaskVisualizer(
                self._vis_dir,
                frames_uint8,
                self.config.visualization_max_frames,
            )

        region_results: Dict[str, Dict[str, object]] = {}
        overall_anomalies: List[Dict[str, object]] = []
        scores: List[float] = []

        for region in self.config.regions:
            masks, coverage = self._build_region_masks(region, frames_uint8, keypoint_sequence)
            if self._combined_visualizer is not None:
                for frame_idx, mask in enumerate(masks):
                    self._combined_visualizer.add_region_mask(frame_idx, region.name, mask)
            detector = RegionTemporalChangeDetector(self.flow_analyzer, region.temporal_config)
            region_result = detector.analyze(video_frames, masks, fps=fps, label=region.name)

            region_metadata = region_result.setdefault("metadata", {})
            region_metadata["mask_coverage"] = [float(np.clip(val, 0.0, 1.0)) for val in coverage]
            region_metadata["region"] = region.name

            region_results[region.name] = region_result
            scores.append(float(region_result.get("score", 1.0)))

            region_anomalies = region_result.get("anomalies", [])
            overall_anomalies.extend(
                {**anomaly, "region": region.name} for anomaly in region_anomalies
            )

            if self._vis_dir and self.config.per_region_visualization:
                region_metadata["visualization_dir"] = str(self._vis_dir / region.name)

        overall_score = float(np.mean(scores)) if scores else 1.0
        metadata: Dict[str, object] = {"regions": list(region_results.keys())}
        
        if self._combined_visualizer is not None:
            combined_output = self._combined_visualizer.save()
            if combined_output is not None:
                metadata["combined_visualization_dir"] = str(combined_output)
            self._combined_visualizer = None

        if simple_output:
            # 简化输出模式：只包含基本信息
            region_summary = {}
            for region_name, region_result in region_results.items():
                region_anomalies = region_result.get("anomalies", [])
                region_summary[region_name] = {
                    "score": float(region_result.get("score", 1.0)),
                    "has_anomaly": len(region_anomalies) > 0,
                    "anomaly_count": len(region_anomalies),
                }
            
            return {
                "score": overall_score,
                "has_anomaly": len(overall_anomalies) > 0,
                "total_anomaly_count": len(overall_anomalies),
                "region_summary": region_summary,
            }
        else:
            # 详细输出模式：包含所有信息
            return {
                "score": overall_score,
                "anomalies": overall_anomalies,
                "regions": region_results,
                "metadata": metadata,
            }

    def _build_region_masks(
        self,
        region: RegionDefinition,
        frames_uint8: Sequence[np.ndarray],
        keypoint_sequence: Sequence[Dict[str, Optional[np.ndarray]]],
    ) -> Tuple[List[Optional[np.ndarray]], List[float]]:
        masks: List[Optional[np.ndarray]] = []
        coverage: List[float] = []
        for idx, (frame_uint8, keypoints) in enumerate(zip(frames_uint8, keypoint_sequence)):
            height, width = frame_uint8.shape[:2]
            mask = self._build_region_mask(region, keypoints, height, width)
            masks.append(mask)
            coverage.append(float(mask.sum()) / float(mask.size) if mask is not None else 0.0)
            if self._vis_dir:
                self._save_mask_visualization(region, frame_uint8, keypoints, mask, idx)
        return masks, coverage

    def _build_region_mask(
        self,
        region: RegionDefinition,
        keypoints: Dict[str, Optional[np.ndarray]],
        height: int,
        width: int,
    ) -> Optional[np.ndarray]:
        polygon = self._extract_region_polygon(region, keypoints, height, width)
        if polygon is None or polygon.size == 0:
            return None

        polygon = polygon.astype(np.int32)
        polygon[..., 0] = np.clip(polygon[..., 0], 0, width - 1)
        polygon[..., 1] = np.clip(polygon[..., 1], 0, height - 1)

        mask_uint8 = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask_uint8, [polygon], 1)

        if region.margin_ratio > 0.0:
            iterations = max(1, int(round(max(height, width) * region.margin_ratio)))
            mask_uint8 = cv2.dilate(mask_uint8, np.ones((3, 3), dtype=np.uint8), iterations=iterations)

        area = int(mask_uint8.sum())
        if area < region.min_area:
            return None

        return mask_uint8.astype(bool)

    def _extract_region_polygon(
        self,
        region: RegionDefinition,
        keypoints: Dict[str, Optional[np.ndarray]],
        height: int,
        width: int,
    ) -> Optional[np.ndarray]:
        group_points = self._get_group_points(region.keypoint_group, keypoints)
        if group_points is None or group_points.shape[0] == 0:
            return None

        points = group_points[:, :2]
        if region.keypoint_indices:
            try:
                points = points[np.array(region.keypoint_indices)]
            except Exception:
                return None

        points = np.clip(points, 0.0, 1.0)
        if np.any(np.isnan(points)):
            return None

        pts_px = np.stack([points[:, 0] * width, points[:, 1] * height], axis=1)

        if region.mask_mode == RegionMaskMode.POLYGON:
            if pts_px.shape[0] < 3:
                return None
            return pts_px.reshape((-1, 1, 2))

        if region.mask_mode == RegionMaskMode.CONVEX_HULL:
            if pts_px.shape[0] < 3:
                return None
            hull = cv2.convexHull(pts_px.astype(np.float32))
            if hull is None or len(hull) < 3:
                return None
            return hull.reshape((-1, 1, 2))

        if region.mask_mode == RegionMaskMode.BOUNDING_BOX:
            x_min, y_min = np.min(pts_px, axis=0)
            x_max, y_max = np.max(pts_px, axis=0)
            polygon = np.array(
                [
                    [[x_min, y_min]],
                    [[x_max, y_min]],
                    [[x_max, y_max]],
                    [[x_min, y_max]],
                ],
                dtype=np.float32,
            )
            return polygon

        return None

    @staticmethod
    def _get_group_points(group: str, keypoints: Dict[str, Optional[np.ndarray]]) -> Optional[np.ndarray]:
        mapping = {
            "face": keypoints.get("face"),
            "left_hand": keypoints.get("left_hand"),
            "right_hand": keypoints.get("right_hand"),
            "pose": keypoints.get("body"),
        }
        if group not in mapping:
            raise ValueError(f"Unsupported keypoint group: {group}")
        return mapping[group]

    @staticmethod
    def _ensure_uint8(frame: np.ndarray) -> np.ndarray:
        if frame.dtype == np.uint8:
            return frame
        return np.clip(frame, 0, 255).astype(np.uint8)

    def _prepare_visualization(self, video_path: Optional[str]) -> None:
        if not self.config.enable_visualization:
            self._vis_dir = None
            self._vis_counters = {}
            return

        base_dir = (
            Path(self.config.visualization_output_dir).expanduser()
            if self.config.visualization_output_dir
            else Path(__file__).resolve().parents[3] / "outputs" / "region_analysis"
        )
        base_dir.mkdir(parents=True, exist_ok=True)
        video_name = Path(video_path).stem if video_path else "video"
        self._vis_dir = (base_dir / video_name).resolve()
        self._vis_dir.mkdir(parents=True, exist_ok=True)
        self._vis_counters = {}

    def _save_mask_visualization(
        self,
        region: RegionDefinition,
        frame: np.ndarray,
        keypoints: Dict[str, Optional[np.ndarray]],
        selected_mask: Optional[np.ndarray],
        frame_idx: int,
    ) -> None:
        if self._vis_dir is None or not self.config.per_region_visualization:
            return

        max_frames = self.config.visualization_max_frames or 0
        counter = self._vis_counters.get(region.name, 0)
        if max_frames > 0 and counter >= max_frames:
            return

        region_dir = self._vis_dir / region.name
        region_dir.mkdir(parents=True, exist_ok=True)

        frame_uint8 = self._ensure_uint8(frame)
        frame_rgb = frame_uint8.copy()
        overlay = frame_rgb.copy()

        if selected_mask is not None:
            mask_bool = selected_mask.astype(bool)
            overlay[mask_bool] = (40, 40, 40)
            mask_uint8 = (mask_bool.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        polygon = self._extract_region_polygon(
            region,
            keypoints,
            height=frame_rgb.shape[0],
            width=frame_rgb.shape[1],
        )
        if polygon is not None:
            cv2.polylines(overlay, [polygon.astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=1)

        blended = cv2.addWeighted(frame_rgb, 0.6, overlay, 0.4, 0)
        cv2.putText(
            blended,
            f"{region.name} frame {frame_idx:04d}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        save_path = region_dir / f"frame_{frame_idx:04d}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        self._vis_counters[region.name] = counter + 1


