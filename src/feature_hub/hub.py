from __future__ import annotations

from typing import Any, Callable

from .cache import FeatureCache

ExtractorFn = Callable[[str, str], Any]


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
        result = self._extractors[feature_name](self.video_path, self.device)
        self._cache.store(feature_name, result)
        return result

    def has_cached(self, feature_name: str) -> bool:
        return self._cache.has(feature_name)

    def available_features(self) -> list[str]:
        return list(self._extractors.keys())

    def clear_cache(self) -> None:
        self._cache.clear()
