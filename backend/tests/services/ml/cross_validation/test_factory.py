"""
cross_validation/factory.py のユニットテスト

infer_timeframe / get_t1_series / create_temporal_cv_splitter をテストします。
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from app.services.ml.cross_validation.factory import (
    create_temporal_cv_splitter,
    get_t1_series,
    infer_timeframe,
)
from app.services.ml.cross_validation.purged_kfold import PurgedKFold


@pytest.fixture
def hourly_index() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=200, freq="h")


@pytest.fixture
def daily_index() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=200, freq="D")


# ---------------------------------------------------------------------------
# infer_timeframe
# ---------------------------------------------------------------------------

class TestInferTimeframe:
    def test_hourly(self, hourly_index):
        assert infer_timeframe(hourly_index) == "1h"

    def test_daily(self, daily_index):
        assert infer_timeframe(daily_index) == "1d"

    def test_4h(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="4h")
        assert infer_timeframe(idx) == "4h"

    def test_15m(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="15min")
        assert infer_timeframe(idx) == "15m"

    def test_30m(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="30min")
        assert infer_timeframe(idx) == "30m"

    def test_single_element(self):
        idx = pd.DatetimeIndex(["2024-01-01"])
        assert infer_timeframe(idx) == "1h"

    def test_empty(self):
        idx = pd.DatetimeIndex([])
        assert infer_timeframe(idx) == "1h"

    def test_unknown_interval(self):
        """未知の間隔でも 1h を返す"""
        idx = pd.date_range("2024-01-01", periods=5, freq="37min")
        result = infer_timeframe(idx)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# get_t1_series
# ---------------------------------------------------------------------------

class TestGetT1Series:
    def test_hourly(self, hourly_index):
        t1 = get_t1_series(hourly_index, horizon_n=12, timeframe="1h")
        assert isinstance(t1, pd.Series)
        assert len(t1) == len(hourly_index)
        assert t1.index.equals(hourly_index)
        # 各値は 12時間後
        expected_delta = pd.Timedelta(hours=12)
        assert t1.iloc[0] == hourly_index[0] + expected_delta

    def test_daily(self, daily_index):
        t1 = get_t1_series(daily_index, horizon_n=5, timeframe="1D")
        assert len(t1) == len(daily_index)

    def test_4h_timeframe(self):
        idx = pd.date_range("2024-01-01", periods=50, freq="4h")
        t1 = get_t1_series(idx, horizon_n=6, timeframe="4h")
        expected = idx[0] + pd.Timedelta(hours=24)
        assert t1.iloc[0] == expected

    def test_minute_timeframe(self):
        idx = pd.date_range("2024-01-01", periods=50, freq="15min")
        t1 = get_t1_series(idx, horizon_n=4, timeframe="15m")
        expected = idx[0] + pd.Timedelta(minutes=60)
        assert t1.iloc[0] == expected

    def test_auto_timeframe(self, hourly_index):
        """timeframe=None でも自動推定される"""
        t1 = get_t1_series(hourly_index, horizon_n=12)
        assert len(t1) == len(hourly_index)


# ---------------------------------------------------------------------------
# create_temporal_cv_splitter
# ---------------------------------------------------------------------------

class TestCreateTemporalCvSplitter:
    def test_kfold(self, hourly_index):
        splitter = create_temporal_cv_splitter("kfold", 5, hourly_index)
        assert isinstance(splitter, KFold)

    def test_stratified_kfold(self, hourly_index):
        splitter = create_temporal_cv_splitter("stratified_kfold", 3, hourly_index)
        assert isinstance(splitter, StratifiedKFold)

    def test_purged_kfold_with_t1(self, hourly_index):
        t1 = get_t1_series(hourly_index, horizon_n=12, timeframe="1h")
        splitter = create_temporal_cv_splitter(
            "purged_kfold", 5, hourly_index, t1=t1
        )
        assert isinstance(splitter, PurgedKFold)

    def test_purged_kfold_with_horizon(self, hourly_index):
        splitter = create_temporal_cv_splitter(
            "purged_kfold", 5, hourly_index, horizon_n=12, timeframe="1h"
        )
        assert isinstance(splitter, PurgedKFold)

    def test_purged_kfold_no_horizon_no_t1_raises(self, hourly_index):
        with pytest.raises(ValueError, match="horizon_n"):
            create_temporal_cv_splitter("purged_kfold", 5, hourly_index)

    def test_unsupported_strategy_raises(self, hourly_index):
        with pytest.raises(ValueError, match="Unsupported"):
            create_temporal_cv_splitter("unknown", 5, hourly_index)

    def test_default_strategy_is_purged(self, hourly_index):
        splitter = create_temporal_cv_splitter(
            None, 5, hourly_index, horizon_n=12, timeframe="1h"
        )
        assert isinstance(splitter, PurgedKFold)

    def test_splits_data(self, hourly_index):
        """分割が正しく行われる"""
        splitter = create_temporal_cv_splitter(
            "kfold", 5, hourly_index
        )
        y = pd.Series(np.random.randint(0, 2, len(hourly_index)), index=hourly_index)
        splits = list(splitter.split(hourly_index, y))
        assert len(splits) == 5
