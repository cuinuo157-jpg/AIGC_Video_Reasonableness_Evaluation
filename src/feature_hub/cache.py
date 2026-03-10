from __future__ import annotations

from typing import Any


class FeatureCache:
    """内存特征缓存，支持按 key 存取特征数据。"""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def store(self, key: str, data: Any) -> None:
        self._store[key] = data

    def get(self, key: str) -> Any | None:
        return self._store.get(key)

    def has(self, key: str) -> bool:
        return key in self._store

    def clear(self) -> None:
        self._store.clear()

    def keys(self) -> list[str]:
        return list(self._store.keys())
