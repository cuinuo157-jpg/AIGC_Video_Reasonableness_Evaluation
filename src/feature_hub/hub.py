from __future__ import annotations

import inspect
from typing import Any, Callable

from .cache import FeatureCache

# 支持两种签名:
#   (video_path, device) -> Any              # 旧版/简单 extractor
#   (video_path, device, hub) -> Any         # 需要依赖其他 extractor 的
ExtractorFn = Callable[..., Any]


class FeatureHub:
    """共享基础特征层：懒加载 + 缓存，避免各维度重复提取特征。"""

    def __init__(self, video_path: str, device: str = "cuda") -> None:
        self.video_path = video_path
        self.device = device
        self._cache = FeatureCache()
        self._extractors: dict[str, ExtractorFn] = {}

    def register_extractor(self, feature_name: str, extractor: ExtractorFn) -> None:
        self._extractors[feature_name] = extractor

    def get(self, feature_name: str) -> Any:
        if self._cache.has(feature_name):
            return self._cache.get(feature_name)
        if feature_name not in self._extractors:
            raise KeyError(
                f"Unknown feature: {feature_name}. "
                f"Available: {list(self._extractors.keys())}"
            )
        fn = self._extractors[feature_name]
        # 自动检测函数是否接受 hub 参数
        sig = inspect.signature(fn)
        if len(sig.parameters) >= 3:
            result = fn(self.video_path, self.device, self)
        else:
            result = fn(self.video_path, self.device)
        self._cache.store(feature_name, result)
        return result

    def has_cached(self, feature_name: str) -> bool:
        return self._cache.has(feature_name)

    def has_extractor(self, feature_name: str) -> bool:
        """检查是否注册了指定名称的特征提取器。"""
        return feature_name in self._extractors

    def available_features(self) -> list[str]:
        return list(self._extractors.keys())

    def clear_cache(self) -> None:
        self._cache.clear()


def create_default_hub(video_path: str, device: str = "cuda") -> FeatureHub:
    """创建预注册所有默认特征提取器的 FeatureHub 实例。"""
    from .extractors.optical_flow import extract_optical_flow
    from .extractors.face_embedding import extract_face_embeddings
    from .extractors.depth import extract_depth_maps
    from .extractors.mediapipe_keypoints import extract_mediapipe_keypoints
    from .extractors.video_frames import extract_video_frames
    from .extractors.camera_compensation import extract_camera_compensation
    from .extractors.keypoint_tracking import extract_keypoint_trajectories
    from .extractors.au_features import extract_au_features

    hub = FeatureHub(video_path, device)
    hub.register_extractor("optical_flow", extract_optical_flow)
    hub.register_extractor("face_embedding", extract_face_embeddings)
    hub.register_extractor("depth", extract_depth_maps)
    hub.register_extractor("keypoints", extract_mediapipe_keypoints)
    hub.register_extractor("video_frames", extract_video_frames)
    hub.register_extractor("camera_compensation", extract_camera_compensation)
    hub.register_extractor("tracking", extract_keypoint_trajectories)
    hub.register_extractor("au_features", extract_au_features)

    # RAFT 光流: RAFT 不可用时内部自动降级为 Farneback
    from .extractors.raft_flow import extract_raft_flow
    hub.register_extractor("raft_flow", extract_raft_flow)

    from .extractors.subject_segmentation import extract_subject_masks
    hub.register_extractor("subject_masks", extract_subject_masks)

    return hub
