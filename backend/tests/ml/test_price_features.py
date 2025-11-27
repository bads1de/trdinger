"""
price_features.pyのテスト
TDDで開発し、リファクタリングの安全性を確保します。
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from app.services.ml.feature_engineering.price_features import PriceFeatureCalculator


@pytest.fixture
def sample_ohlcv_data():
    """サンプルOHLCVデータを生成"""
    dates = pd.date_range(start=datetime(2023, 1, 1), periods=1000, freq="1h")

    np.random.seed(42)
    base_price = 50000

    data = []
    for i, date in enumerate(dates):
        change = np.random.randn() * 100
        base_price += change
        high = base_price + abs(np.random.randn()) * 50
        low = base_price - abs(np.random.randn()) * 50
        volume = np.random.randint(100, 10000)

        data.append(
            {
                "timestamp": date,
                "open": base_price - change / 2,
                "high": high,
                "low": low,
                "close": base_price,
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    return df


def test_price_feature_calculator_initialization():
    """PriceFeatureCalculatorの初期化をテスト"""
    calculator = PriceFeatureCalculator()
    assert calculator is not None
    assert hasattr(calculator, "calculate_features")


def test_calculate_price_features(sample_ohlcv_data):
    """価格特徴量のテスト"""
    calculator = PriceFeatureCalculator()
    lookback_periods = {"short_ma": 10, "long_ma": 50}

    result = calculator.calculate_price_features(sample_ohlcv_data, lookback_periods)

    expected_features = [
        "Price_Change_1",
        "Price_Change_5",
        "Price_Change_20",
        "Body_Size",
        "Lower_Shadow",
    ]

    for feature in expected_features:
        assert feature in result.columns, f"Missing feature: {feature}"
        assert not result[feature].isna().all(), f"Feature {feature} is all NaN"


def test_calculate_volatility_features(sample_ohlcv_data):
    """ボラティリティ特徴量のテスト"""
    calculator = PriceFeatureCalculator()
    lookback_periods = {"volatility": 20}

    result = calculator.calculate_volatility_features(
        sample_ohlcv_data, lookback_periods
    )

    expected_features = ["ATR_20"]

    for feature in expected_features:
        assert feature in result.columns, f"Missing feature: {feature}"
        assert not result[feature].isna().all(), f"Feature {feature} is all NaN"


def test_calculate_volume_features(sample_ohlcv_data):
    """出来高特徴量のテスト"""
    calculator = PriceFeatureCalculator()
    lookback_periods = {"volume": 20}

    result = calculator.calculate_volume_features(sample_ohlcv_data, lookback_periods)

    expected_features = [
        "Volume_MA_20",
        "Price_Volume_Trend",
        "VWAP",
        "VWAP_Deviation",
        "Volume_Trend",
    ]

    for feature in expected_features:
        assert feature in result.columns, f"Missing feature: {feature}"
        assert not result[feature].isna().all(), f"Feature {feature} is all NaN"


def test_calculate_features_integration(sample_ohlcv_data):
    """統合的な特徴量計算のテスト"""
    calculator = PriceFeatureCalculator()
    config = {
        "lookback_periods": {
            "short_ma": 10,
            "long_ma": 50,
            "volatility": 20,
            "volume": 20,
        }
    }

    result = calculator.calculate_features(sample_ohlcv_data, config)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_ohlcv_data)

    # 全ての特徴量が含まれているか確認
    expected_features = calculator.get_feature_names()
    for feature in expected_features:
        assert feature in result.columns, f"Missing feature: {feature}"
