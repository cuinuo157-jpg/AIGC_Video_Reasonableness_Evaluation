import numpy as np

from src.feature_hub.cache import FeatureCache


def test_cache_store_and_retrieve():
    cache = FeatureCache()
    data = np.random.rand(10, 20)
    cache.store("optical_flow", data)
    result = cache.get("optical_flow")
    assert result is not None
    np.testing.assert_array_equal(result, data)


def test_cache_miss_returns_none():
    cache = FeatureCache()
    assert cache.get("nonexistent") is None


def test_cache_clear():
    cache = FeatureCache()
    cache.store("test", np.array([1, 2, 3]))
    cache.clear()
    assert cache.get("test") is None


def test_cache_has():
    cache = FeatureCache()
    assert not cache.has("key")
    cache.store("key", "value")
    assert cache.has("key")
