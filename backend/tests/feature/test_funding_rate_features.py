"""
ファンディングレート特徴量計算のテスト
"""

import numpy as np
import pandas as pd
import pytest

from app.services.ml.feature_engineering.funding_rate_features import (
    FundingRateFeatureCalculator,
)


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """1時間足のOHLCVサンプルデータ（1000時間）"""
    periods = 1000
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="1h")
    close_prices = 40000 + np.linspace(0, 5000, periods) + np.random.randn(periods) * 500

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close_prices * 0.999,
        "high": close_prices * 1.002,
        "low": close_prices * 0.998,
        "close": close_prices,
        "volume": np.random.uniform(100, 1000, periods),
    })


@pytest.fixture
def sample_funding_data() -> pd.DataFrame:
    """8時間ごとのファンディングレートサンプルデータ"""
    periods = 135
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="8h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "funding_rate": np.random.uniform(-0.0005, 0.001, periods),
    })


@pytest.fixture
def calculator() -> FundingRateFeatureCalculator:
    return FundingRateFeatureCalculator()


class TestFundingRateFeatureCalculator:
    """FundingRateFeatureCalculatorのテストクラス"""

    def test_initialization(self, calculator: FundingRateFeatureCalculator):
        assert calculator.settlement_interval == 8
        assert calculator.baseline_rate == 0.0001

    def test_calculate_features_basic(
        self, calculator, sample_ohlcv_data, sample_funding_data
    ):
        result = calculator.calculate_features(sample_ohlcv_data, sample_funding_data)
        assert result.index.name == "timestamp"
        assert len(result) == len(sample_ohlcv_data)

    def test_time_cycle_features(self, calculator, sample_ohlcv_data, sample_funding_data):
        result = calculator.calculate_features(sample_ohlcv_data, sample_funding_data)
        assert "fr_cycle_sin" in result.columns
        assert "fr_cycle_cos" in result.columns

    def test_basic_rate_features(self, calculator, sample_ohlcv_data, sample_funding_data):
        result = calculator.calculate_features(sample_ohlcv_data, sample_funding_data)
        assert "fr_bps" in result.columns
        assert "fr_dev" in result.columns
        assert "fr_lag_3p" in result.columns

    def test_momentum_features(self, calculator, sample_ohlcv_data, sample_funding_data):
        result = calculator.calculate_features(sample_ohlcv_data, sample_funding_data)
        assert "fr_ema_3p" in result.columns

    def test_regime_classification(self, calculator, sample_ohlcv_data):
        ts = pd.date_range("2024-01-01", periods=5, freq="8h")
        funding = pd.DataFrame({
            "timestamp": ts,
            "funding_rate": [-0.00015, -0.00005, 0.0003, 0.0008, 0.002],
        })
        result = calculator.calculate_features(sample_ohlcv_data, funding)
        assert "fr_regime" in result.columns
        assert result["fr_regime"].dropna().isin([-2, -1, 0, 1, 2]).all()

    def test_price_interaction_features(self, calculator, sample_ohlcv_data, sample_funding_data):
        result = calculator.calculate_features(sample_ohlcv_data, sample_funding_data)
        assert "fr_price_corr" in result.columns

    def test_missing_value_handling(self, calculator, sample_ohlcv_data):
        ts = pd.date_range("2024-01-01", periods=10, freq="8h")
        funding = pd.DataFrame({
            "timestamp": ts,
            "funding_rate": [0.0001, np.nan, 0.0002, np.nan, 0.0003, 0.0004, np.nan, 0.0005, 0.0006, 0.0007],
        })
        result = calculator.calculate_features(sample_ohlcv_data, funding)
        assert "fr_bps" in result.columns
        assert not result["fr_bps"].isna().all()

    def test_data_frequency_mismatch(self, calculator, sample_ohlcv_data, sample_funding_data):
        result = calculator.calculate_features(sample_ohlcv_data, sample_funding_data)
        assert len(result) == len(sample_ohlcv_data)
        assert "fr_bps" in result.columns

    def test_feature_output_shape(self, calculator, sample_ohlcv_data, sample_funding_data):
        result = calculator.calculate_features(sample_ohlcv_data, sample_funding_data)
        expected = [
            "fr_bps", "fr_dev", "fr_lag_3p", "fr_cycle_sin", "fr_cycle_cos",
            "fr_ema_3p", "fr_regime", "fr_price_corr",
            "fr_zscore_72h", "fr_zscore_168h", "fr_extreme", "fr_direction",
        ]
        for feature in expected:
            assert feature in result.columns

    def test_extreme_funding_rates(self, calculator, sample_ohlcv_data):
        ts = pd.date_range("2024-01-01", periods=5, freq="8h")
        funding = pd.DataFrame({
            "timestamp": ts,
            "funding_rate": [-0.01, 0.005, -0.005, 0.01, 0.0],
        })
        result = calculator.calculate_features(sample_ohlcv_data, funding)
        assert "fr_bps" in result.columns

    def test_single_funding_rate_record(self, calculator, sample_ohlcv_data):
        funding = pd.DataFrame({
            "timestamp": [pd.Timestamp("2024-01-01")],
            "funding_rate": [0.0001],
        })
        result = calculator.calculate_features(sample_ohlcv_data, funding)
        assert "fr_bps" in result.columns

    def test_numerical_stability(self, calculator, sample_ohlcv_data, sample_funding_data):
        result = calculator.calculate_features(sample_ohlcv_data, sample_funding_data)
        num_cols = result.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            assert np.isinf(result[col]).sum() == 0
            assert (result[col].isna().sum() / len(result)) < 0.5