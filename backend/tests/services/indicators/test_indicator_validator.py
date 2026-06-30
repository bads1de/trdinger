"""
indicator_validator モジュールのユニットテスト
"""

import numpy as np
import pandas as pd
import pytest

from app.services.indicators.indicator_validator import IndicatorValidator


@pytest.fixture
def validator():
    return IndicatorValidator()


@pytest.fixture
def sample_df():
    index = pd.date_range("2024-01-01", periods=50, freq="h")
    return pd.DataFrame(
        {"Close": range(50), "Open": range(50), "High": range(50), "Low": range(50)},
        index=index,
    )


class TestIndicatorValidator:
    def test_resolve_column_name_exact_match(self, validator, sample_df):
        result = validator.resolve_column_name(sample_df, "Close")
        assert result == "Close"

    def test_resolve_column_name_case_insensitive(self, validator, sample_df):
        result = validator.resolve_column_name(sample_df, "close")
        assert result == "Close"

    def test_resolve_column_name_upper(self, validator, sample_df):
        result = validator.resolve_column_name(sample_df, "CLOSE")
        assert result == "Close"

    def test_resolve_column_name_with_underscore(self, validator, sample_df):
        result = validator.resolve_column_name(sample_df, "Close_")
        assert result == "Close"

    def test_resolve_column_name_not_found(self, validator, sample_df):
        result = validator.resolve_column_name(sample_df, "nonexistent")
        assert result is None

    def test_resolve_column_name_none(self, validator, sample_df):
        result = validator.resolve_column_name(sample_df, None)
        assert result is None

    def test_create_nan_result_single(self, validator, sample_df):
        config = {"function": "RSI", "returns": "single"}
        result = validator.create_nan_result(sample_df, config)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_df)
        assert np.isnan(result).all()

    def test_create_nan_result_multiple(self, validator, sample_df):
        config = {
            "function": "MACD",
            "returns": "multiple",
            "return_cols": ["macd", "signal", "hist"],
        }
        result = validator.create_nan_result(sample_df, config)
        assert isinstance(result, tuple)
        assert len(result) == 3
        for arr in result:
            assert len(arr) == len(sample_df)
            assert np.isnan(arr).all()

    def test_basic_validation_valid(self, validator, sample_df):
        config = {
            "function": "RSI",
            "data_column": "close",
            "multi_column": False,
        }
        params = {"length": 14}
        assert validator.basic_validation(sample_df, config, params) is True

    def test_basic_validation_short_data(self, validator):
        index = pd.date_range("2024-01-01", periods=3, freq="h")
        short_df = pd.DataFrame(
            {
                "Close": [1, 2, 3],
                "Open": [1, 2, 3],
                "High": [1, 2, 3],
                "Low": [1, 2, 3],
            },
            index=index,
        )
        config = {
            "function": "RSI",
            "data_column": "close",
            "multi_column": False,
        }
        params = {"length": 100}
        assert validator.basic_validation(short_df, config, params) is False

    def test_basic_validation_missing_column(self, validator, sample_df):
        config = {
            "function": "RSI",
            "data_column": "nonexistent",
            "multi_column": False,
        }
        params = {"length": 14}
        assert validator.basic_validation(sample_df, config, params) is False
