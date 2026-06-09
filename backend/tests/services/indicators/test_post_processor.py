"""
post_processor モジュールのユニットテスト
"""

import numpy as np
import pandas as pd
import pytest

from app.services.indicators.post_processor import PostProcessor


@pytest.fixture
def processor():
    return PostProcessor()


@pytest.fixture
def sample_df():
    index = pd.date_range("2024-01-01", periods=50, freq="h")
    return pd.DataFrame({"close": range(50)}, index=index)


class TestPostProcessor:
    def test_post_process_series_single(self, processor, sample_df):
        series = pd.Series(range(50), index=sample_df.index, name="result")
        config = {"returns": "single"}
        result = processor.post_process(series, config, sample_df)
        assert isinstance(result, np.ndarray)
        assert len(result) == 50

    def test_post_process_series_multi(self, processor, sample_df):
        series = pd.Series(range(50), index=sample_df.index, name="result")
        config = {"returns": "multiple", "return_cols": ["a"]}
        result = processor.post_process(series, config, sample_df)
        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_post_process_dataframe_single(self, processor, sample_df):
        df = pd.DataFrame({"a": range(50)}, index=sample_df.index)
        config = {"returns": "single"}
        result = processor.post_process(df, config, sample_df)
        assert isinstance(result, np.ndarray)

    def test_post_process_dataframe_multi(self, processor, sample_df):
        df = pd.DataFrame({"a": range(50), "b": range(50, 100)}, index=sample_df.index)
        config = {"returns": "multiple", "return_cols": ["a", "b"]}
        result = processor.post_process(df, config, sample_df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_post_process_tuple_first_element(self, processor, sample_df):
        series = pd.Series(range(50), index=sample_df.index)
        tup = (series,)
        config = {"returns": "single"}
        result = processor.post_process(tup, config, sample_df)
        assert isinstance(result, np.ndarray)

    def test_post_process_ndarray(self, processor, sample_df):
        arr = np.arange(50, dtype=float)
        config = {"returns": "single"}
        result = processor.post_process(arr, config, sample_df)
        assert isinstance(result, np.ndarray)

    def test_post_process_reindex(self, processor, sample_df):
        short_series = pd.Series(range(30), index=pd.date_range("2024-01-01", periods=30, freq="h"))
        config = {"returns": "single"}
        result = processor.post_process(short_series, config, sample_df)
        assert isinstance(result, np.ndarray)

    def test_post_process_dataframe_return_cols_partial_match(self, processor, sample_df):
        df = pd.DataFrame({"MACD_12_26_9": range(50), "MACDs_12_26_9": range(50)}, index=sample_df.index)
        config = {"returns": "multiple", "return_cols": ["MACD"]}
        result = processor.post_process(df, config, sample_df)
        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_post_process_dataframe_return_cols_missing(self, processor, sample_df):
        df = pd.DataFrame({"a": range(50)}, index=sample_df.index)
        config = {"returns": "multiple", "return_cols": ["nonexistent"]}
        result = processor.post_process(df, config, sample_df)
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert np.isnan(result[0]).all()
