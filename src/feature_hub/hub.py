from __future__ import annotations

import inspect
import threading
from dataclasses import dataclass
from typing import Any, Callable

from .cache import FeatureCache

# 支持两种签名:
#   (video_path, device) -> Any              # 旧版/简单 extractor
#   (video_path, device, hub) -> Any         # 需要依赖其他 extractor 的
ExtractorFn = Callable[..., Any]


@dataclass(frozen=True)
class VideoProcessingConfig:
    """共享视频预处理配置。"""

    sample_stride: int = 1
    max_frames: int | None = None
    max_side: int | None = None


class FeatureHub:
    """共享基础特征层：懒加载 + 缓存，避免各维度重复提取特征。"""

    def __init__(
        self,
        video_path: str,
        device: str = "cuda",
        video_config: VideoProcessingConfig | None = None,
    ) -> None:
        self.video_path = video_path
        self.device = device
        self.video_config = video_config or VideoProcessingConfig()
        self._cache = FeatureCache()
        self._extractors: dict[str, ExtractorFn] = {}
        self._lock = threading.RLock()
        self._inflight: dict[str, threading.Event] = {}

    def register_extractor(self, feature_name: str, extractor: ExtractorFn) -> None:
        self._extractors[feature_name] = extractor

    def get(self, feature_name: str) -> Any:
        with self._lock:
            if self._cache.has(feature_name):
                return self._cache.get(feature_name)
            if feature_name not in self._extractors:
                raise KeyError(
                    f"Unknown feature: {feature_name}. "
                    f"Available: {list(self._extractors.keys())}"
                )
            wait_event = self._inflight.get(feature_name)
            if wait_event is None:
                wait_event = threading.Event()
                self._inflight[feature_name] = wait_event
                should_compute = True
            else:
                should_compute = False

        if not should_compute:
            wait_event.wait()
            with self._lock:
                if self._cache.has(feature_name):
                    return self._cache.get(feature_name)
            raise RuntimeError(f"Feature extraction failed: {feature_name}")

        fn = self._extractors[feature_name]
        try:
            sig = inspect.signature(fn)
            if len(sig.parameters) >= 3:
                result = fn(self.video_path, self.device, self)
            else:
                result = fn(self.video_path, self.device)
        except Exception:
            with self._lock:
                self._inflight.pop(feature_name, None)
                wait_event.set()
            raise

        with self._lock:
            self._cache.store(feature_name, result)
            self._inflight.pop(feature_name, None)
            wait_event.set()
        return result

    def has_cached(self, feature_name: str) -> bool:
        return self._cache.has(feature_name)

    def has_extractor(self, feature_name: str) -> bool:
        """检查是否注册了指定名称的特征提取器。"""
        return feature_name in self._extractors

    def available_features(self) -> list[str]:
        return list(self._extractors.keys())

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()
            self._inflight.clear()


def create_default_hub(
    video_path: str,
    device: str = "cuda",
    video_config: VideoProcessingConfig | None = None,
) -> FeatureHub:
    """创建预注册所有默认特征提取器的 FeatureHub 实例。"""
    from .extractors.optical_flow import extract_optical_flow
    from .extractors.face_embedding import extract_face_embeddings
    from .extractors.depth import extract_depth_maps
    from .extractors.mediapipe_keypoints import extract_mediapipe_keypoints
    from .extractors.video_frames import extract_video_frames
    from .extractors.camera_compensation import extract_camera_compensation
    from .extractors.keypoint_tracking import extract_keypoint_trajectories
    from .extractors.cotracker_tracking import extract_cotracker_trajectories
    from .extractors.iris_tracking import extract_iris_tracking
    from .extractors.au_features import extract_au_features

    hub = FeatureHub(video_path, device, video_config=video_config)
    hub.register_extractor("optical_flow", extract_optical_flow)
    hub.register_extractor("face_embedding", extract_face_embeddings)
    hub.register_extractor("depth", extract_depth_maps)
    hub.register_extractor("keypoints", extract_mediapipe_keypoints)
    hub.register_extractor("video_frames", extract_video_frames)
    hub.register_extractor("camera_compensation", extract_camera_compensation)
    # 默认 tracking: CoTracker（失败时在 extractor 内自动回退到 keypoint tracking）
    hub.register_extractor("tracking", extract_cotracker_trajectories)
    # 保留显式轻量轨迹 extractor（MediaPipe 关键点）
    hub.register_extractor("tracking_keypoints", extract_keypoint_trajectories)
    hub.register_extractor("iris_tracking", extract_iris_tracking)
    hub.register_extractor("au_features", extract_au_features)

    # RAFT 光流: RAFT 不可用时内部自动降级为 Farneback
    from .extractors.raft_flow import extract_raft_flow
    hub.register_extractor("raft_flow", extract_raft_flow)

    from .extractors.subject_segmentation import extract_subject_masks
    hub.register_extractor("subject_masks", extract_subject_masks)

    return hub
