"""
extract_tuple_result ヘルパーのユニットテスト

DataFrameからtupleへの変換、nan_resultフォールバックなどの共通パターンをテストする。
"""

import numpy as np
import pandas as pd
import pytest

from app.services.indicators.data_validation import (
    extract_tuple_result,
    nan_result_for,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"MACD_12_26_9": [1.0, 2.0, 3.0], "MACDs_12_26_9": [0.5, 1.5, 2.5], "MACDh_12_26_9": [0.5, 0.5, 0.5]},
        index=pd.date_range("2024-01-01", periods=3, freq="h"),
    )


@pytest.fixture
def sample_series() -> pd.Series:
    return pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        index=pd.date_range("2024-01-01", periods=5, freq="h"),
        name="close",
    )


# ---------------------------------------------------------------------------
# extract_tuple_result
# ---------------------------------------------------------------------------

class TestExtractTupleResult:
    def test_tuple_passthrough(self):
        """tuple入力はそのまま返す"""
        t = (pd.Series([1, 2]), pd.Series([3, 4]))
        assert extract_tuple_result(t, 2) is t

    def test_dataframe_to_tuple_by_index(self, sample_df):
        """DataFrameをindex指定でtupleに変換"""
        result = extract_tuple_result(sample_df, 3, by_index=True)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert list(result[0]) == [1.0, 2.0, 3.0]

    def test_dataframe_to_tuple_by_names(self, sample_df):
        """DataFrameをcolumn名指定でtupleに変換"""
        result = extract_tuple_result(
            sample_df, 3, column_names=["MACD_12_26_9", "MACDs_12_26_9"]
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert list(result[0]) == [1.0, 2.0, 3.0]

    def test_dataframe_to_tuple_to_numpy(self, sample_df):
        """to_numpy=Trueでndarrayを返す"""
        result = extract_tuple_result(sample_df, 3, to_numpy=True)
        assert isinstance(result, tuple)
        assert isinstance(result[0], np.ndarray)

    def test_fallback_on_key_error(self, sample_df):
        """存在しないcolumn名はfallbackを返す"""
        fallback = (pd.Series([99, 99, 99]), pd.Series([88, 88, 88]))
        result = extract_tuple_result(
            sample_df, 2, column_names=["NONEXISTENT"], fallback_factory=lambda: fallback
        )
        assert result is fallback

    def test_fallback_on_index_error(self, sample_df):
        """index範囲外はfallbackを返す"""
        fallback = (pd.Series([99, 99, 99]),)
        result = extract_tuple_result(
            sample_df, 5, by_index=True, fallback_factory=lambda: fallback
        )
        assert result is fallback


# ---------------------------------------------------------------------------
# nan_result_for
# ---------------------------------------------------------------------------

class TestNanResultFor:
    def test_single_series(self, sample_series):
        """単一SeriesからNaN bundleを生成"""
        result = nan_result_for(sample_series, 3)
        assert isinstance(result, tuple)
        assert len(result) == 3
        for s in result:
            assert len(s) == len(sample_series)
            assert s.isna().all()

    def test_single_series_numpy(self, sample_series):
        """numpy=Trueでndarrayを返す"""
        result = nan_result_for(sample_series, 3, to_numpy=True)
        assert isinstance(result, tuple)
        assert isinstance(result[0], np.ndarray)
        assert np.isnan(result[0]).all()

    def test_from_dict(self, sample_series):
        """dict入力から参照Seriesを抽出"""
        result = nan_result_for({"close": sample_series}, 2)
        assert len(result) == 2

    def test_from_tuple(self, sample_series):
        """tuple入力はそのまま返す（NaN結果として扱う）"""
        nan_tuple = tuple(pd.Series([np.nan] * 5) for _ in range(3))
        result = nan_result_for(nan_tuple, 3)
        assert result is nan_tuple
