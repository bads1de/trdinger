"""
label_cache モジュールのユニットテスト
"""

import pandas as pd
import pytest

from app.services.ml.label_generation.label_cache import LabelCache, ThresholdMethod
from tests.helpers.data import make_ohlcv_df


@pytest.fixture
def sample_ohlcv():
    return make_ohlcv_df(periods=200)


class TestThresholdMethod:
    def test_triple_barrier(self):
        assert ThresholdMethod.TRIPLE_BARRIER.value == "triple_barrier"

    def test_trend_scanning(self):
        assert ThresholdMethod.TREND_SCANNING.value == "trend_scanning"


class TestLabelCache:
    def test_initialization(self, sample_ohlcv):
        cache = LabelCache(sample_ohlcv)
        assert cache.ohlcv_df is sample_ohlcv
        assert cache.cache == {}
        assert cache.hit_count == 0
        assert cache.miss_count == 0

    def test_get_labels_trend_scanning(self, sample_ohlcv):
        cache = LabelCache(sample_ohlcv)
        labels = cache.get_labels(
            horizon_n=4,
            threshold_method="TREND_SCANNING",
            threshold=0.0001,
            timeframe="1h",
        )
        assert isinstance(labels, pd.Series)

    def test_cache_hit(self, sample_ohlcv):
        cache = LabelCache(sample_ohlcv)
        labels1 = cache.get_labels(
            horizon_n=4,
            threshold_method="TREND_SCANNING",
            threshold=0.002,
        )
        labels2 = cache.get_labels(
            horizon_n=4,
            threshold_method="TREND_SCANNING",
            threshold=0.002,
        )
        assert cache.hit_count == 1
        assert cache.miss_count == 1
        assert labels1 is labels2

    def test_different_params_different_cache(self, sample_ohlcv):
        cache = LabelCache(sample_ohlcv)
        cache.get_labels(
            horizon_n=4, threshold_method="TREND_SCANNING", threshold=0.002
        )
        cache.get_labels(
            horizon_n=8, threshold_method="TREND_SCANNING", threshold=0.002
        )
        assert cache.miss_count == 2
        assert cache.hit_count == 0

    def test_t_events_bypasses_cache(self, sample_ohlcv):
        cache = LabelCache(sample_ohlcv)
        t_events = pd.DatetimeIndex([sample_ohlcv.index[10], sample_ohlcv.index[20]])
        cache.get_labels(
            horizon_n=4,
            threshold_method="TREND_SCANNING",
            threshold=0.002,
            t_events=t_events,
        )
        assert cache.miss_count == 0
        assert cache.hit_count == 0

    def test_invalid_threshold_method(self, sample_ohlcv):
        cache = LabelCache(sample_ohlcv)
        with pytest.raises(ValueError, match="無効な閾値計算方法"):
            cache.get_labels(
                horizon_n=4,
                threshold_method="INVALID",
                threshold=0.002,
            )

    def test_get_hit_rate(self, sample_ohlcv):
        cache = LabelCache(sample_ohlcv)
        assert cache.get_hit_rate() == 0.0

        cache.get_labels(
            horizon_n=4, threshold_method="TREND_SCANNING", threshold=0.002
        )
        cache.get_labels(
            horizon_n=4, threshold_method="TREND_SCANNING", threshold=0.002
        )
        assert cache.get_hit_rate() == 50.0

    def test_clear(self, sample_ohlcv):
        cache = LabelCache(sample_ohlcv)
        cache.get_labels(
            horizon_n=4, threshold_method="TREND_SCANNING", threshold=0.002
        )
        cache.clear()
        assert cache.cache == {}
        assert cache.hit_count == 0
        assert cache.miss_count == 0

    def test_get_stats(self, sample_ohlcv):
        cache = LabelCache(sample_ohlcv)
        stats = cache.get_stats()
        assert "hit_count" in stats
        assert "miss_count" in stats
        assert "cache_size" in stats
        assert "hit_rate_pct" in stats
