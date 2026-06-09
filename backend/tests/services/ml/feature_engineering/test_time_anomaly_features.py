"""
time_anomaly_features モジュールのユニットテスト
"""

import numpy as np
import pandas as pd
import pytest

from app.services.ml.feature_engineering.time_anomaly_features import (
    TimeAnomalyFeatures,
)


@pytest.fixture
def sample_df_with_datetime_index() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=100, freq="h")
    return pd.DataFrame(
        {
            "close": 100.0 + np.cumsum(np.random.randn(100) * 0.5),
            "volume": np.random.rand(100) * 1000,
        },
        index=index,
    )


@pytest.fixture
def sample_df_with_timestamp_column() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=50, freq="h")
    return pd.DataFrame(
        {
            "timestamp": index,
            "close": 100.0 + np.cumsum(np.random.randn(50) * 0.5),
            "volume": np.random.rand(50) * 1000,
        }
    )


class TestTimeAnomalyFeatures:
    def test_calculate_features_basic(self, sample_df_with_datetime_index):
        calc = TimeAnomalyFeatures()
        result = calc.calculate_features(sample_df_with_datetime_index)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df_with_datetime_index)
        assert "time_hour_sin" in result.columns
        assert "time_hour_cos" in result.columns
        assert "time_day_sin" in result.columns
        assert "time_day_cos" in result.columns

    def test_market_sessions(self, sample_df_with_datetime_index):
        calc = TimeAnomalyFeatures()
        result = calc.calculate_features(sample_df_with_datetime_index)

        assert "time_session_asia" in result.columns
        assert "time_session_europe" in result.columns
        assert "time_session_us" in result.columns
        assert "time_session_overlap" in result.columns

        assert result["time_session_asia"].isin([0, 1]).all()
        assert result["time_session_europe"].isin([0, 1]).all()
        assert result["time_session_us"].isin([0, 1]).all()

    def test_calendar_anomalies(self, sample_df_with_datetime_index):
        calc = TimeAnomalyFeatures()
        result = calc.calculate_features(sample_df_with_datetime_index)

        assert "time_is_weekend" in result.columns
        assert "time_is_month_end" in result.columns
        assert "time_is_monday" in result.columns
        assert "time_is_friday" in result.columns
        assert "time_is_daily_close" in result.columns

        assert result["time_is_weekend"].isin([0, 1]).all()
        assert result["time_is_monday"].isin([0, 1]).all()

    def test_interaction_features_with_volume(self, sample_df_with_datetime_index):
        calc = TimeAnomalyFeatures()
        result = calc.calculate_features(sample_df_with_datetime_index)

        assert "time_interaction_vol_asia" in result.columns
        assert "time_interaction_vol_europe" in result.columns
        assert "time_interaction_vol_us" in result.columns

    def test_interaction_features_with_close(self, sample_df_with_datetime_index):
        calc = TimeAnomalyFeatures()
        result = calc.calculate_features(sample_df_with_datetime_index)

        assert "time_interaction_volatility_us" in result.columns
        assert "time_interaction_volatility_overlap" in result.columns

    def test_adaptive_volatility(self, sample_df_with_datetime_index):
        calc = TimeAnomalyFeatures()
        result = calc.calculate_features(sample_df_with_datetime_index)

        assert "time_adaptive_vol_ratio" in result.columns
        assert np.isfinite(result["time_adaptive_vol_ratio"].iloc[24:]).all()

    def test_microstructure_proxy(self, sample_df_with_datetime_index):
        calc = TimeAnomalyFeatures()
        result = calc.calculate_features(sample_df_with_datetime_index)

        assert "time_micro_illiquidity" in result.columns

    def test_session_elapsed_time(self, sample_df_with_datetime_index):
        calc = TimeAnomalyFeatures()
        result = calc.calculate_features(sample_df_with_datetime_index)

        assert "time_since_tokyo" in result.columns
        assert "time_since_london" in result.columns
        assert "time_since_ny" in result.columns

    def test_with_timestamp_column(self, sample_df_with_timestamp_column):
        calc = TimeAnomalyFeatures()
        result = calc.calculate_features(sample_df_with_timestamp_column)

        assert isinstance(result, pd.DataFrame)
        assert "time_hour_sin" in result.columns

    def test_no_datetime_index_no_timestamp_column(self):
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0], "volume": [100, 200, 300]})
        calc = TimeAnomalyFeatures()
        result = calc.calculate_features(df)
        assert result.equals(df)

    def test_empty_dataframe(self):
        df = pd.DataFrame(
            {"close": pd.Series(dtype=float), "volume": pd.Series(dtype=float)},
            index=pd.DatetimeIndex([]),
        )
        calc = TimeAnomalyFeatures()
        result = calc.calculate_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_no_volume_column(self):
        index = pd.date_range("2024-01-01", periods=30, freq="h")
        df = pd.DataFrame({"close": 100.0 + np.arange(30)}, index=index)
        calc = TimeAnomalyFeatures()
        result = calc.calculate_features(df)
        assert "time_interaction_vol_asia" not in result.columns
