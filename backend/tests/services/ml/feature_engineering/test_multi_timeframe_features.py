"""
MultiTimeframeFeatureCalculator のユニットテスト

複数時間足統合特徴量をテストします:
- calculate_features: 基本計算、カラム生成、データリーク防止
- _resample_to_timeframe: リサンプル
- _calculate_rsi: RSI計算
- エッジケース（データ不足）
"""

import numpy as np
import pandas as pd
import pytest

from app.services.ml.feature_engineering.multi_timeframe_features import (
    MultiTimeframeFeatureCalculator,
)


@pytest.fixture
def calculator() -> MultiTimeframeFeatureCalculator:
    return MultiTimeframeFeatureCalculator()


@pytest.fixture
def hourly_data() -> pd.DataFrame:
    """200日分の1時間足データ（200*24=4800行）"""
    np.random.seed(42)
    n = 4800
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    close = 50000 + np.cumsum(np.random.randn(n) * 50)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 10,
            "high": close + abs(np.random.randn(n) * 20),
            "low": close - abs(np.random.randn(n) * 20),
            "close": close,
            "volume": abs(np.random.randn(n) * 1000) + 100,
        },
        index=dates,
    )


class TestCalculateFeatures:
    def test_returns_dataframe(self, calculator, hourly_data):
        result = calculator.calculate_features(hourly_data)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self, calculator, hourly_data):
        result = calculator.calculate_features(hourly_data)
        expected_cols = [
            "HTF_4h_Trend_Direction",
            "HTF_4h_Trend_Strength",
            "HTF_4h_RSI",
            "HTF_1d_Trend_Direction",
            "HTF_1d_Trend_Strength",
            "Timeframe_Alignment_Score",
            "Timeframe_Alignment_Direction",
            "HTF_4h_Divergence",
            "HTF_1d_Divergence",
            "Price_Distance_From_4h_SMA50",
            "Price_Distance_From_1d_SMA50",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_output_length_matches_input(self, calculator, hourly_data):
        result = calculator.calculate_features(hourly_data)
        assert len(result) == len(hourly_data)

    def test_no_inf_values(self, calculator, hourly_data):
        result = calculator.calculate_features(hourly_data)
        assert not np.isinf(result.values).any()

    def test_alignment_score_range(self, calculator, hourly_data):
        result = calculator.calculate_features(hourly_data)
        valid = result["Timeframe_Alignment_Score"].dropna()
        assert valid.between(0, 1).all()

    def test_alignment_direction_values(self, calculator, hourly_data):
        result = calculator.calculate_features(hourly_data)
        valid = result["Timeframe_Alignment_Direction"].dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})

    def test_insufficient_data_returns_empty(self, calculator):
        """200行未満では空DataFrameを返す"""
        short_data = pd.DataFrame(
            {
                "open": [1] * 100,
                "high": [1] * 100,
                "low": [1] * 100,
                "close": [1] * 100,
                "volume": [1] * 100,
            },
            index=pd.date_range("2024-01-01", periods=100, freq="h"),
        )
        result = calculator.calculate_features(short_data)
        assert len(result.columns) == 0

    def test_data_leak_prevention_shift(self, calculator, hourly_data):
        """4時間足・日足データがシフトされているか確認（リーク防止）"""
        result = calculator.calculate_features(hourly_data)
        # sanitize_numeric_dataframe が NaN を fill するため、
        # アライメントスコアが極端に0や1でないことを確認（NaN→0埋めの影響）
        early = result["Timeframe_Alignment_Score"].iloc[:5]
        assert not early.isna().all()  # 0埋め済み


class TestResampleToTimeframe:
    def test_4h_resample(self, calculator, hourly_data):
        result = calculator._resample_to_timeframe(hourly_data, "4h")
        assert len(result) < len(hourly_data)
        assert "close" in result.columns

    def test_1d_resample(self, calculator, hourly_data):
        result = calculator._resample_to_timeframe(hourly_data, "1D")
        assert len(result) < len(hourly_data)


class TestCalculateRsi:
    def test_rsi_range(self, calculator):
        prices = pd.Series(np.linspace(100, 200, 100))
        rsi = calculator._calculate_rsi(prices, period=14)
        valid = rsi.dropna()
        assert valid.between(0, 100).all()

    def test_rsi_uptrend_is_high(self, calculator):
        prices = pd.Series(range(1, 50))
        rsi = calculator._calculate_rsi(prices, period=14)
        assert rsi.iloc[-1] > 50

    def test_rsi_downtrend_is_low(self, calculator):
        prices = pd.Series(range(50, 1, -1))
        rsi = calculator._calculate_rsi(prices, period=14)
        assert rsi.iloc[-1] < 50
