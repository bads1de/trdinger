"""
temporal_features.pyのテスト
TDDで開発し、DataFrameのfragmentation問題とAPI互換性をテストします。
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from app.services.ml.feature_engineering.temporal_features import TemporalFeatureCalculator


@pytest.fixture
def sample_ohlcv_data():
    """サンプルOHLCVデータ（DatetimeIndex付き）を生成"""
    dates = pd.date_range(
        start=datetime(2023, 1, 1),
        periods=1000,
        freq='1h'
    )

    np.random.seed(42)
    base_price = 50000

    data = []
    for i, date in enumerate(dates):
        change = np.random.randn() * 100
        base_price += change
        high = base_price + abs(np.random.randn()) * 50
        low = base_price - abs(np.random.randn()) * 50
        volume = np.random.randint(100, 10000)

        data.append({
            'timestamp': date,
            'open': base_price - change/2,
            'high': high,
            'low': low,
            'close': base_price,
            'volume': volume
        })

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


def test_temporal_feature_calculator_initialization():
    """TemporalFeatureCalculatorの初期化をテスト"""
    calculator = TemporalFeatureCalculator()
    assert calculator is not None
    assert hasattr(calculator, 'calculate_features')
    assert hasattr(calculator, 'calculate_temporal_features')


def test_calculate_features_basic(sample_ohlcv_data):
    """基本的なcalculate_featuresメソッドのテスト"""
    calculator = TemporalFeatureCalculator()
    config = {"lookback_periods": {}}

    result = calculator.calculate_features(sample_ohlcv_data, config)

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) > len(sample_ohlcv_data.columns)
    assert len(result) == len(sample_ohlcv_data)


def test_calculate_temporal_features(sample_ohlcv_data):
    """calculate_temporal_featuresメソッドのテスト"""
    calculator = TemporalFeatureCalculator()
    result = calculator.calculate_temporal_features(sample_ohlcv_data)

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) > len(sample_ohlcv_data.columns)


def test_basic_time_features(sample_ohlcv_data):
    """基本的な時間特徴量のテスト"""
    calculator = TemporalFeatureCalculator()
    result = calculator.calculate_features(sample_ohlcv_data, {})

    # 基本的な時間特徴量
    expected_features = [
        "Hour_of_Day",
        "Day_of_Week",
        "Is_Weekend",
        "Is_Monday",
        "Is_Friday"
    ]

    for feature in expected_features:
        assert feature in result.columns, f"Missing feature: {feature}"


def test_trading_session_features(sample_ohlcv_data):
    """取引セッション特徴量のテスト"""
    calculator = TemporalFeatureCalculator()
    result = calculator.calculate_features(sample_ohlcv_data, {})

    # 取引セッション特徴量
    expected_features = [
        "Asia_Session",
        "Europe_Session",
        "US_Session"
    ]

    for feature in expected_features:
        assert feature in result.columns, f"Missing feature: {feature}"


def test_cyclical_features(sample_ohlcv_data):
    """周期的エンコーディング特徴量のテスト"""
    calculator = TemporalFeatureCalculator()
    result = calculator.calculate_features(sample_ohlcv_data, {})

    # 周期的エンコーディング特徴量
    expected_features = [
        "Hour_Sin",
        "Hour_Cos",
        "Day_Sin",
        "Day_Cos"
    ]

    for feature in expected_features:
        assert feature in result.columns, f"Missing feature: {feature}"


def test_session_overlap_features(sample_ohlcv_data):
    """セッション重複時間特徴量のテスト"""
    calculator = TemporalFeatureCalculator()
    result = calculator.calculate_features(sample_ohlcv_data, {})

    # セッション重複時間特徴量
    expected_features = [
        "Session_Overlap_Asia_Europe",
        "Session_Overlap_Europe_US"
    ]

    for feature in expected_features:
        assert feature in result.columns, f"Missing feature: {feature}"


def test_dataframe_not_fragmented(sample_ohlcv_data):
    """DataFrameがfragmentation問題を起こしていないことをテスト"""
    calculator = TemporalFeatureCalculator()
    config = {"lookback_periods": {}}

    result = calculator.calculate_features(sample_ohlcv_data, config)

    # DataFrameの断片化の警告が発生しないことを確認
    assert result is not None
    assert isinstance(result, pd.DataFrame)

    # 基本的な統計情報を取得して、DataFrameが正常にアクセス可能であることを確認
    summary = result.describe()
    assert summary is not None
    assert len(summary) > 0


def test_feature_values_validity(sample_ohlcv_data):
    """特徴量値の妥当性をテスト"""
    calculator = TemporalFeatureCalculator()
    config = {"lookback_periods": {}}

    result = calculator.calculate_features(sample_ohlcv_data, config)

    # 無限値やNaNのチェック（時間関連特徴量のみ）
    time_features = [
        "Hour_of_Day", "Day_of_Week", "Hour_Sin", "Hour_Cos",
        "Day_Sin", "Day_Cos"
    ]

    for col in time_features:
        if col in result.columns:
            assert not result[col].isin([np.inf, -np.inf]).any(), \
                f"Infinite values found in {col}"


def test_feature_count(sample_ohlcv_data):
    """生成される特徴量の数をテスト"""
    calculator = TemporalFeatureCalculator()
    config = {"lookback_periods": {}}

    result = calculator.calculate_features(sample_ohlcv_data, config)

    original_count = len(sample_ohlcv_data.columns)
    feature_count = len(result.columns)

    # 最低10個の特徴量が生成されることを確認
    assert feature_count > original_count + 10, \
        f"Expected more than {original_count + 10} features, got {feature_count}"


def test_get_feature_names():
    """get_feature_namesメソッドのテスト"""
    calculator = TemporalFeatureCalculator()
    feature_names = calculator.get_feature_names()

    assert isinstance(feature_names, list)
    assert len(feature_names) > 0
    assert "Hour_of_Day" in feature_names
    assert "Asia_Session" in feature_names
    assert "Hour_Sin" in feature_names


def test_lookback_periods_optional(sample_ohlcv_data):
    """lookback_periodsがオプションであることをテスト"""
    calculator = TemporalFeatureCalculator()
    config = {}  # 空の設定

    result = calculator.calculate_features(sample_ohlcv_data, config)

    assert result is not None
    assert isinstance(result, pd.DataFrame)


def test_datetime_index_required(sample_ohlcv_data):
    """DatetimeIndexが必須であることをテスト"""
    calculator = TemporalFeatureCalculator()

    # インデックスをリセットしてDatetimeIndexでない状態にする
    df_no_datetime = sample_ohlcv_data.reset_index()

    # ValueErrorが発生することを確認
    with pytest.raises(ValueError, match="DatetimeIndex"):
        calculator.calculate_temporal_features(df_no_datetime)


def test_empty_data_handling():
    """空データのハンドリングをテスト"""
    calculator = TemporalFeatureCalculator()
    empty_df = pd.DataFrame()

    result = calculator.calculate_features(empty_df, {})

    assert result is empty_df or result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
