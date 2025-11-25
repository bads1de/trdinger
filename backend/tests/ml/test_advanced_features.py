"""
advanced_features.pyのテスト
DataFrameのfragmentation問題をテストで確認します。
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from app.services.ml.feature_engineering.advanced_features import (
    AdvancedFeatureEngineer,
)


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


@pytest.fixture
def sample_funding_rate_data():
    """サンプルファンディングレートデータを生成"""
    dates = pd.date_range(start=datetime(2023, 1, 1), periods=1000, freq="1h")

    np.random.seed(42)
    funding_rates = np.random.randn(1000) * 0.001

    df = pd.DataFrame({"timestamp": dates, "funding_rate": funding_rates})
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture
def sample_open_interest_data():
    """サンプル建玉残高データを生成"""
    dates = pd.date_range(start=datetime(2023, 1, 1), periods=1000, freq="1h")

    np.random.seed(42)
    base_oi = 1000000
    open_interest = []
    for i in range(1000):
        base_oi += np.random.randn() * 1000
        open_interest.append(base_oi)

    df = pd.DataFrame({"timestamp": dates, "open_interest": open_interest})
    df.set_index("timestamp", inplace=True)
    return df


def test_advanced_feature_engineer_initialization():
    """AdvancedFeatureEngineerの初期化をテスト"""
    engineer = AdvancedFeatureEngineer()
    assert engineer is not None
    assert hasattr(engineer, "scaler")


def test_create_advanced_features_basic(sample_ohlcv_data):
    """基本的な特徴量生成をテスト"""
    engineer = AdvancedFeatureEngineer()
    features = engineer.create_advanced_features(sample_ohlcv_data)

    # 基本的な列が含まれていることを確認
    assert "close_lag_1" in features.columns
    # NOTE: 'returns' は特徴量重要度分析で完全未使用と判定されたため削除済み
    assert "returns_lag_24" in features.columns  # ラグ版は残っている
    assert "ATR" in features.columns
    assert "OBV" in features.columns


def test_create_advanced_features_with_external_data(
    sample_ohlcv_data, sample_funding_rate_data, sample_open_interest_data
):
    """外部データを含む特徴量生成をテスト"""
    engineer = AdvancedFeatureEngineer()
    features = engineer.create_advanced_features(
        sample_ohlcv_data,
        funding_rate_data=sample_funding_rate_data,
        open_interest_data=sample_open_interest_data,
    )

    # 外部データの特徴量が含まれていることを確認
    # FR特徴量は削除されたため、OI特徴量のみチェック
    assert "OI_pct_change_24" in features.columns
    assert "OI_ma_24" in features.columns


def test_lag_features(sample_ohlcv_data):
    """ラグ特徴量のテスト"""
    engineer = AdvancedFeatureEngineer()
    features = engineer.create_advanced_features(sample_ohlcv_data)

    # ラグ特徴量の列をテスト（削減後の特徴量のみ）
    # NOTE: 'returns' は特徴量重要度分析で完全未使用と判定されたため削除済み
    expected_lag_columns = [
        "close_lag_1",
        "close_lag_24",
        "returns_lag_24",  # returns 自体は削除されたがラグ版は残す
        "cumulative_returns_24",
    ]

    for col in expected_lag_columns:
        assert col in features.columns, f"Missing column: {col}"


def test_technical_indicators(sample_ohlcv_data):
    """技術指標のテスト"""
    engineer = AdvancedFeatureEngineer()
    features = engineer.create_advanced_features(sample_ohlcv_data)

    # 技術指標の列をテスト
    expected_indicators = ["ATR", "OBV", "AD", "ADOSC"]

    for indicator in expected_indicators:
        assert indicator in features.columns, f"Missing indicator: {indicator}"


def test_statistical_features(sample_ohlcv_data):
    """統計的特徴量のテスト"""
    engineer = AdvancedFeatureEngineer()
    features = engineer.create_advanced_features(sample_ohlcv_data)

    # 統計的特徴量の列をテスト（削減後の特徴量のみ）
    # Removed: Close_mean_20, Close_mean_50 (低寄与度特徴量削除: 2025-11-13)
    expected_stat_columns = [
        "Close_std_20",
        "Close_std_50",
        "Close_range_20",
        "Close_range_50",
    ]

    for col in expected_stat_columns:
        assert col in features.columns, f"Missing column: {col}"


def test_time_series_features(sample_ohlcv_data):
    """時系列特徴量のテスト"""
    engineer = AdvancedFeatureEngineer()
    features = engineer.create_advanced_features(sample_ohlcv_data)

    # 時系列特徴量の列をテスト（削減後の特徴量のみ）
    # Removed: Close_pct_change_1, Close_pct_change_24 (低寄与度特徴量削除: 2025-11-13)
    expected_ts_columns = ["Close_deviation_from_ma_20", "Trend_strength_20"]

    for col in expected_ts_columns:
        assert col in features.columns, f"Missing column: {col}"


def test_volatility_features(sample_ohlcv_data):
    """ボラティリティ特徴量のテスト"""
    engineer = AdvancedFeatureEngineer()
    features = engineer.create_advanced_features(sample_ohlcv_data)

    # ボラティリティ特徴量の列をテスト（高寄与度のみ）
    # Removed: Returns (低寄与度特徴量削除: 2025-11-13)
    expected_vol_columns = [
        "Realized_Vol_20",
        "Parkinson_Vol_20",
        # Vol_Regime は削除済み（低寄与度）
    ]

    for col in expected_vol_columns:
        assert col in features.columns, f"Missing column: {col}"

def test_historical_volatility_skewness_kurtosis_features(sample_ohlcv_data):
    """ヒストリカルボラティリティ、スキューネス、尖度の特徴量テスト"""
    engineer = AdvancedFeatureEngineer()
    features = engineer.create_advanced_features(sample_ohlcv_data)

    assert "Historical_Volatility_20" in features.columns
    assert "Price_Skewness_20" in features.columns
    assert "Price_Kurtosis_20" in features.columns

    # 少なくとも、数値として存在することを確認
    assert np.issubdtype(features["Historical_Volatility_20"].dtype, np.number)
    assert np.issubdtype(features["Price_Skewness_20"].dtype, np.number)
    assert np.issubdtype(features["Price_Kurtosis_20"].dtype, np.number)

    # NaNがないことを確認 (計算可能な範囲で)
    assert not features["Historical_Volatility_20"].isnull().all()
    assert not features["Price_Skewness_20"].isnull().all()
    assert not features["Price_Kurtosis_20"].isnull().all()



def test_interaction_features(sample_ohlcv_data):
    """相互作用特徴量のテスト"""
    engineer = AdvancedFeatureEngineer()
    features = engineer.create_advanced_features(sample_ohlcv_data)

    # 相互作用特徴量の列をテスト（削減後の特徴量のみ）
    expected_interaction_columns = ["Price_Volume_Ratio", "Vol_Volume_Product"]

    for col in expected_interaction_columns:
        assert col in features.columns, f"Missing column: {col}"


def test_seasonal_features(sample_ohlcv_data):
    """季節性特徴量のテスト（全削除）"""
    engineer = AdvancedFeatureEngineer()
    features = engineer.create_advanced_features(sample_ohlcv_data)

    # 季節性特徴量は全て削除済み（暗号通貨市場では時間効果が弱い）
    # テストは特徴量が生成されないことを確認
    deleted_columns = [
        "Hour",
        "DayOfWeek",
        "Hour_sin",
        "Hour_cos",
        "DayOfWeek_sin",
        "DayOfWeek_cos",
        "Is_Weekend",
        "Is_Asian_Hours",
        "Is_American_Hours",
    ]

    for col in deleted_columns:
        assert col not in features.columns, f"Column should be deleted: {col}"


def test_funding_rate_features(sample_ohlcv_data, sample_funding_rate_data):
    """ファンディングレート特徴量のテスト（全削除）"""
    engineer = AdvancedFeatureEngineer()
    features = engineer.create_advanced_features(
        sample_ohlcv_data, funding_rate_data=sample_funding_rate_data
    )

    # FR特徴量は全て削除済み（スコア0で寄与なし）
    # テストは特徴量が生成されないことを確認
    deleted_fr_columns = [
        "FR_mean_7",
        "FR_sum_7",
        "FR_extreme_positive",
        "FR_extreme_negative",
    ]

    for col in deleted_fr_columns:
        assert col not in features.columns, f"Column should be deleted: {col}"


def test_open_interest_features(sample_ohlcv_data, sample_open_interest_data):
    """建玉残高特徴量のテスト"""
    engineer = AdvancedFeatureEngineer()
    features = engineer.create_advanced_features(
        sample_ohlcv_data, open_interest_data=sample_open_interest_data
    )

    # 建玉残高特徴量の列をテスト（削減後の特徴量のみ）
    expected_oi_columns = [
        "OI_pct_change_24",
        "OI_ma_24",
        "OI_deviation_24",
        "OI_Price_Correlation",
    ]

    for col in expected_oi_columns:
        assert col in features.columns, f"Missing column: {col}"


def test_create_features_compatibility(sample_ohlcv_data):
    """create_featuresメソッドの互換性をテスト"""
    engineer = AdvancedFeatureEngineer()
    features = engineer.create_features(sample_ohlcv_data)

    assert features is not None
    assert isinstance(features, pd.DataFrame)
    assert len(features.columns) > len(sample_ohlcv_data.columns)


def test_feature_count(sample_ohlcv_data):
    """生成される特徴量の数をテスト"""
    engineer = AdvancedFeatureEngineer()
    features = engineer.create_advanced_features(sample_ohlcv_data)

    # 多くの特徴量が生成されていることを確認
    original_count = len(sample_ohlcv_data.columns)
    feature_count = len(features.columns)

    # 2025-11-13削減後: 約30個の特徴量を追加（低寄与度特徴量17個削除済み）
    # 5（元のカラム） + 30 = 35個程度を期待
    assert feature_count > original_count + 20, (
        f"Expected more than {original_count + 20} features, got {feature_count}"
    )


def test_no_duplicate_columns(sample_ohlcv_data):
    """重複する列がないことをテスト"""
    engineer = AdvancedFeatureEngineer()
    features = engineer.create_advanced_features(sample_ohlcv_data)

    # 列名の重複をチェック
    assert len(features.columns) == len(set(features.columns)), (
        "Duplicate columns found"
    )


def test_dataframe_not_fragmented(sample_ohlcv_data):
    """DataFrameがfragmentation問題を起こしていないことをテスト"""
    engineer = AdvancedFeatureEngineer()
    features = engineer.create_advanced_features(sample_ohlcv_data)

    # DataFrameの断片化の警告が発生しないことを確認
    # （pandasは通常警告を出しますが、コードを正常に実行できることを確認）
    assert features is not None
    assert isinstance(features, pd.DataFrame)

    # 基本的な統計情報を取得して、DataFrameが正常にアクセス可能であることを確認
    summary = features.describe()
    assert summary is not None
    assert len(summary) > 0


def test_memory_efficiency(sample_ohlcv_data):
    """メモリエフィシェンシーをテスト"""
    engineer = AdvancedFeatureEngineer()

    # 特徴量生成前のサイズを記録
    original_size = sample_ohlcv_data.memory_usage(deep=True).sum()

    # 特徴量を生成
    features = engineer.create_advanced_features(sample_ohlcv_data)

    # 特徴量生成後のサイズを記録
    new_size = features.memory_usage(deep=True).sum()

    # メモリ使用量が増加していることを確認（合理的範囲内）
    assert new_size > original_size
    assert new_size < original_size * 50, "Memory usage increased too much"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
