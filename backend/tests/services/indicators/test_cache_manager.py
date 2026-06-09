"""
cache_manager モジュールのユニットテスト
"""

import pandas as pd
import pytest

from app.services.indicators.cache_manager import IndicatorCacheManager


@pytest.fixture
def cache_manager():
    return IndicatorCacheManager(maxsize=100)


@pytest.fixture
def sample_df():
    index = pd.date_range("2024-01-01", periods=50, freq="h")
    return pd.DataFrame(
        {"close": range(50), "volume": range(50, 100)},
        index=index,
    )


class TestIndicatorCacheManager:
    def test_initialization(self, cache_manager):
        assert cache_manager._calculation_cache.maxsize == 100
        assert cache_manager._cache_hits == 0
        assert cache_manager._cache_misses == 0

    def test_make_cache_key(self, cache_manager, sample_df):
        key = cache_manager.make_cache_key("sma", {"length": 20}, sample_df)
        assert key is not None
        assert key[0] == "sma"

    def test_same_data_same_key(self, cache_manager, sample_df):
        key1 = cache_manager.make_cache_key("sma", {"length": 20}, sample_df)
        key2 = cache_manager.make_cache_key("sma", {"length": 20}, sample_df)
        assert key1 == key2

    def test_different_params_different_key(self, cache_manager, sample_df):
        key1 = cache_manager.make_cache_key("sma", {"length": 20}, sample_df)
        key2 = cache_manager.make_cache_key("sma", {"length": 50}, sample_df)
        assert key1 != key2

    def test_different_data_different_key(self, cache_manager, sample_df):
        key1 = cache_manager.make_cache_key("sma", {"length": 20}, sample_df)
        df2 = sample_df.copy()
        df2["close"] = df2["close"] + 10
        key2 = cache_manager.make_cache_key("sma", {"length": 20}, df2)
        assert key1 != key2

    def test_cache_and_retrieve(self, cache_manager, sample_df):
        key = cache_manager.make_cache_key("sma", {"length": 20}, sample_df)
        cache_manager.cache_result(key, "test_result")
        result = cache_manager.get_cached_result(key)
        assert result == "test_result"

    def test_cache_miss(self, cache_manager, sample_df):
        key = cache_manager.make_cache_key("sma", {"length": 20}, sample_df)
        result = cache_manager.get_cached_result(key)
        assert result is None
        assert cache_manager._cache_misses == 1
        assert cache_manager._cache_hits == 0

    def test_cache_hit(self, cache_manager, sample_df):
        key = cache_manager.make_cache_key("sma", {"length": 20}, sample_df)
        cache_manager.cache_result(key, "result")
        cache_manager.get_cached_result(key)
        assert cache_manager._cache_hits == 1

    def test_none_key_returns_none(self, cache_manager):
        assert cache_manager.get_cached_result(None) is None

    def test_none_result_not_cached(self, cache_manager, sample_df):
        key = cache_manager.make_cache_key("sma", {"length": 20}, sample_df)
        cache_manager.cache_result(key, None)
        assert cache_manager.get_cached_result(key) is None

    def test_clear_cache(self, cache_manager, sample_df):
        key = cache_manager.make_cache_key("sma", {"length": 20}, sample_df)
        cache_manager.cache_result(key, "result")
        cache_manager.clear_cache()

        assert len(cache_manager._calculation_cache) == 0
        assert cache_manager._cache_hits == 0
        assert cache_manager._cache_misses == 0

    def test_cache_statistics(self, cache_manager, sample_df):
        key = cache_manager.make_cache_key("sma", {"length": 20}, sample_df)
        cache_manager.cache_result(key, "result")
        cache_manager.get_cached_result(key)

        stats = cache_manager.get_cache_statistics()
        assert stats["cache_size"] == 1
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 0
        assert stats["hit_rate"] == 1.0

    def test_cache_statistics_empty(self, cache_manager):
        stats = cache_manager.get_cache_statistics()
        assert stats["hit_rate"] == 0.0

    def test_empty_df_key(self, cache_manager):
        empty_df = pd.DataFrame()
        key = cache_manager.make_cache_key("sma", {"length": 20}, empty_df)
        assert key is not None
        assert key[2] == ("empty",)
