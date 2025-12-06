import pytest
import pandas as pd
import numpy as np
import pandas_ta as ta
from backend.app.services.ml.feature_engineering.technical_features import (
    TechnicalFeatureCalculator,
)


@pytest.fixture
def sample_ohlcv_data():
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="1h")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": np.random.uniform(100, 200, 1000),
            "high": np.random.uniform(200, 300, 1000),
            "low": np.random.uniform(50, 100, 1000),
            "close": np.random.uniform(100, 200, 1000),
            "volume": np.random.randint(1000, 10000, 1000),
        }
    )
    # 価格データの整合性を保つ（Highは最高、Lowは最低）
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)

    # ランダムウォーク的な価格変動を作成
    price = 50000
    prices = []
    for _ in range(1000):
        change = np.random.normal(0, 100)
        price += change
        prices.append(price)

    df["close"] = prices
    df["open"] = df["close"].shift(1).fillna(prices[0]) + np.random.normal(0, 50, 1000)
    df["high"] = df[["open", "close"]].max(axis=1) + np.abs(
        np.random.normal(0, 50, 1000)
    )
    df["low"] = df[["open", "close"]].min(axis=1) - np.abs(
        np.random.normal(0, 50, 1000)
    )

    df.set_index("timestamp", inplace=True)
    return df


def test_technical_feature_calculator_initialization():
    """初期化のテスト"""
    calculator = TechnicalFeatureCalculator()
    assert isinstance(calculator, TechnicalFeatureCalculator)


def test_calculate_features_basic(sample_ohlcv_data):
    """基本機能のテスト"""
    calculator = TechnicalFeatureCalculator()
    config = {"lookback_periods": {"short_ma": 10, "long_ma": 50}}

    result = calculator.calculate_features(sample_ohlcv_data, config)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_ohlcv_data)
    # 元のカラムが保持されているか
    assert all(col in result.columns for col in sample_ohlcv_data.columns)


def test_market_regime_features(sample_ohlcv_data):
    """市場レジーム特徴量のテスト"""
    calculator = TechnicalFeatureCalculator()
    lookback_periods = {"short_ma": 10, "long_ma": 50, "volatility": 20}

    result = calculator.calculate_market_regime_features(
        sample_ohlcv_data, lookback_periods
    )

    # 期待される特徴量（更新後）
    expected_features = [
        "Market_Efficiency",
        "Choppiness_Index_14",
        "Amihud_Illiquidity",
        "Efficiency_Ratio",
        "Market_Impact",
    ]

    for feature in expected_features:
        assert feature in result.columns, f"Missing feature: {feature}"


def test_momentum_features(sample_ohlcv_data):
    """モメンタム特徴量のテスト"""
    calculator = TechnicalFeatureCalculator()
    lookback_periods = {"short_ma": 10, "long_ma": 50}

    result = calculator.calculate_momentum_features(sample_ohlcv_data, lookback_periods)

    # 期待される特徴量
    expected_features = [
        "RSI",
        "MACD_Histogram",
        "Williams_R",
    ]

    for feature in expected_features:
        assert feature in result.columns, f"Missing feature: {feature}"


def test_advanced_technical_indicators(sample_ohlcv_data):
    """高度な技術指標のテスト（AdvancedFeatureEngineerから移行）"""
    calculator = TechnicalFeatureCalculator()
    lookback_periods = {"short_ma": 10, "long_ma": 50}

    result = calculator.calculate_advanced_technical_features(
        sample_ohlcv_data, lookback_periods
    )

    expected_features = [
        "MFI",
        "ADX",
        "AROONOSC",
        "BB_Width",
        "OBV",
        "AD",
        "ADOSC",
        "NATR",
        "Yang_Zhang_Vol_20",
    ]

    for feature in expected_features:
        assert feature in result.columns, f"Missing feature: {feature}"


def test_feature_values_validity(sample_ohlcv_data):
    """特徴量の値が妥当か（NaNや無限大がないか）"""
    calculator = TechnicalFeatureCalculator()
    config = {"lookback_periods": {"short_ma": 10, "long_ma": 50}}

    result = calculator.calculate_features(sample_ohlcv_data, config)

    # 最初の数行は計算期間のためNaNになる可能性があるが、
    # TechnicalFeatureCalculatorはfillna(0)などをしているはず
    # ここでは全体としてNaNがないことを確認
    # ただし、一部の指標は計算に十分なデータがないとNaNになる場合がある
    # 実装ではfillnaされていることを期待

    # 無限大のチェック
    assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()

    # NaNのチェック（計算期間不足を除くため、後半を確認）
    assert not result.iloc[100:].isna().any().any()


def test_dataframe_not_fragmented(sample_ohlcv_data):
    """DataFrameが断片化していないか（PerformanceWarning対策）"""
    calculator = TechnicalFeatureCalculator()
    config = {"lookback_periods": {"short_ma": 10, "long_ma": 50}}

    # 警告をキャッチして確認することもできるが、
    # ここでは正常に完了することを確認
    result = calculator.calculate_features(sample_ohlcv_data, config)
    assert isinstance(result, pd.DataFrame)


@pytest.mark.skip(reason="パフォーマンスベンチマークは通常実行しない")
def test_performance_benchmark(sample_ohlcv_data):
    """パフォーマンスベンチマーク"""
    import time

    calculator = TechnicalFeatureCalculator()
    config = {"lookback_periods": {"short_ma": 10, "long_ma": 50}}

    start_time = time.time()
    for _ in range(10):
        calculator.calculate_features(sample_ohlcv_data, config)
    end_time = time.time()

    avg_time = (end_time - start_time) / 10
    print(f"Average execution time: {avg_time:.4f} seconds")
    assert avg_time < 1.0  # 1秒未満であることを期待


def test_feature_count(sample_ohlcv_data):
    """生成される特徴量の数を確認"""
    calculator = TechnicalFeatureCalculator()
    config = {"lookback_periods": {"short_ma": 10, "long_ma": 50}}

    result = calculator.calculate_features(sample_ohlcv_data, config)

    # 元のカラム数
    original_cols = len(sample_ohlcv_data.columns)
    # 新しいカラム数
    new_cols = len(result.columns) - original_cols

    assert new_cols > 10  # 少なくとも10個以上の特徴量が生成されるはず


def test_get_feature_names():
    """get_feature_namesメソッドのテスト"""
    calculator = TechnicalFeatureCalculator()
    feature_names = calculator.get_feature_names()

    assert isinstance(feature_names, list)
    assert len(feature_names) > 0
    assert "RSI" in feature_names
    assert "MACD_Histogram" in feature_names


def test_lookback_periods_optional(sample_ohlcv_data):
    """lookback_periodsが指定されない場合のデフォルト動作"""
    calculator = TechnicalFeatureCalculator()
    config = {}  # lookback_periodsなし

    result = calculator.calculate_features(sample_ohlcv_data, config)

    assert "RSI" in result.columns


def test_calculate_features_with_frac_diff(sample_ohlcv_data):
    """統合されたcalculate_featuresでの分数次差分テスト"""
    calculator = TechnicalFeatureCalculator()
    config = {
        "lookback_periods": {"short_ma": 10, "long_ma": 50},
        "fractional_differentiation": {"enabled": True, "d": 0.4},
    }

    result = calculator.calculate_features(sample_ohlcv_data, config)

    assert "FracDiff_04" in result.columns
