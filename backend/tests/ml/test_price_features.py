import pytest
import pandas as pd
import numpy as np
from backend.app.services.ml.feature_engineering.price_features import (
    PriceFeatureCalculator,
)


@pytest.fixture
def sample_ohlcv_data():
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="1h")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": np.random.uniform(50000, 51000, 1000),
            "high": np.random.uniform(51000, 52000, 1000),
            "low": np.random.uniform(49000, 50000, 1000),
            "close": np.random.uniform(50000, 51000, 1000),
            "volume": np.random.randint(1000, 10000, 1000),
        }
    )
    df.set_index("timestamp", inplace=True)
    return df


def test_price_feature_calculator_initialization():
    """初期化のテスト"""
    calculator = PriceFeatureCalculator()
    assert isinstance(calculator, PriceFeatureCalculator)


def test_calculate_price_features(sample_ohlcv_data):
    """価格特徴量のテスト"""
    calculator = PriceFeatureCalculator()
    lookback_periods = {"short_ma": 10, "long_ma": 50}

    result = calculator.calculate_price_features(sample_ohlcv_data, lookback_periods)

    expected_features = [
        "Price_Volume_Trend",
    ]

    for feature in expected_features:
        assert feature in result.columns, f"Missing feature: {feature}"


def test_calculate_statistical_features(sample_ohlcv_data):
    """統計的特徴量のテスト（AdvancedFeatureEngineerから移行）"""
    calculator = PriceFeatureCalculator()
    lookback_periods = {}

    result = calculator.calculate_statistical_features(
        sample_ohlcv_data, lookback_periods
    )

    expected_features = [
        "Close_range_20",
        "Historical_Volatility_20",
        "Price_Skewness_20",
    ]

    for feature in expected_features:
        assert feature in result.columns, f"Missing feature: {feature}"


def test_calculate_time_series_features(sample_ohlcv_data):
    """時系列特徴量のテスト（AdvancedFeatureEngineerから移行）"""
    calculator = PriceFeatureCalculator()
    lookback_periods = {}

    result = calculator.calculate_time_series_features(
        sample_ohlcv_data, lookback_periods
    )

    expected_features = [
        "Trend_strength_20",
    ]

    for feature in expected_features:
        assert feature in result.columns, f"Missing feature: {feature}"


def test_calculate_volatility_features(sample_ohlcv_data):
    """ボラティリティ特徴量のテスト"""
    calculator = PriceFeatureCalculator()
    lookback_periods = {"volatility": 20}

    result = calculator.calculate_volatility_features(
        sample_ohlcv_data, lookback_periods
    )

    expected_features = ["Parkinson_Vol_20"]

    for feature in expected_features:
        assert feature in result.columns, f"Missing feature: {feature}"


def test_calculate_features_integration(sample_ohlcv_data):
    """統合メソッドのテスト"""
    calculator = PriceFeatureCalculator()
    config = {"lookback_periods": {"short_ma": 10, "long_ma": 50}}

    result = calculator.calculate_features(sample_ohlcv_data, config)

    # 全ての特徴量が含まれているか確認
    expected_features = [
        "Price_Volume_Trend",
        "Close_range_20",
        "Historical_Volatility_20",
        "Price_Skewness_20",
        "Trend_strength_20",
        "Parkinson_Vol_20",
    ]

    for feature in expected_features:
        assert feature in result.columns, f"Missing feature: {feature}"
