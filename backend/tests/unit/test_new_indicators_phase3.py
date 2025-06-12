#!/usr/bin/env python3
"""
Phase 3 新規追加指標のテスト
BOP, PPO, MIDPOINT, MIDPRICE, TRIMA指標のテスト
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from app.core.services.indicators.momentum_indicators import (
    BOPIndicator,
    PPOIndicator,
    get_momentum_indicator,
)
from app.core.services.indicators.trend_indicators import (
    MIDPOINTIndicator,
    MIDPRICEIndicator,
    TRIMAIndicator,
    get_trend_indicator,
)
from app.core.services.indicators.adapters.base_adapter import TALibCalculationError


class TestPhase3NewIndicators:
    """Phase 3 新規追加指標のテストクラス"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のOHLCVデータを生成"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")

        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)
        close_prices = base_price * np.exp(np.cumsum(returns))

        return pd.DataFrame(
            {
                "open": close_prices * (1 + np.random.normal(0, 0.001, 100)),
                "high": close_prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
                "low": close_prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
                "close": close_prices,
                "volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )

    def test_bop_indicator(self, sample_data):
        """BOP (Balance Of Power) 指標のテスト"""
        indicator = BOPIndicator()

        # 基本計算テスト
        result = indicator.calculate(sample_data, 1)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert result.name == "BOP"

        # 値の範囲テスト（-1から1の範囲）
        valid_values = result.dropna()
        assert all(valid_values >= -1)
        assert all(valid_values <= 1)

        # 説明テスト
        assert "Balance Of Power" in indicator.get_description()

        # サポート期間テスト
        assert indicator.supported_periods == [1]

    def test_ppo_indicator(self, sample_data):
        """PPO (Percentage Price Oscillator) 指標のテスト"""
        indicator = PPOIndicator()

        # 基本計算テスト
        result = indicator.calculate(sample_data, 12, slow_period=26)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert "PPO_12_26" in result.name

        # 異なるパラメータでのテスト
        result2 = indicator.calculate(sample_data, 10, slow_period=20, matype=0)
        assert "PPO_10_20" in result2.name

        # 説明テスト
        assert "Percentage Price Oscillator" in indicator.get_description()

        # サポート期間テスト
        assert 12 in indicator.supported_periods
        assert 26 in indicator.supported_periods

    def test_midpoint_indicator(self, sample_data):
        """MIDPOINT (MidPoint over period) 指標のテスト"""
        indicator = MIDPOINTIndicator()

        # 基本計算テスト
        result = indicator.calculate(sample_data, 14)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert result.name == "MIDPOINT_14"

        # 異なる期間でのテスト
        result2 = indicator.calculate(sample_data, 20)
        assert result2.name == "MIDPOINT_20"

        # 説明テスト
        assert "MidPoint over period" in indicator.get_description()

        # サポート期間テスト
        assert 14 in indicator.supported_periods
        assert 20 in indicator.supported_periods

    def test_midprice_indicator(self, sample_data):
        """MIDPRICE (Midpoint Price over period) 指標のテスト"""
        indicator = MIDPRICEIndicator()

        # 基本計算テスト
        result = indicator.calculate(sample_data, 14)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert result.name == "MIDPRICE_14"

        # 異なる期間でのテスト
        result2 = indicator.calculate(sample_data, 20)
        assert result2.name == "MIDPRICE_20"

        # 説明テスト
        assert "Midpoint Price over period" in indicator.get_description()

        # サポート期間テスト
        assert 14 in indicator.supported_periods
        assert 20 in indicator.supported_periods

    def test_trima_indicator(self, sample_data):
        """TRIMA (Triangular Moving Average) 指標のテスト"""
        indicator = TRIMAIndicator()

        # 基本計算テスト
        result = indicator.calculate(sample_data, 30)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert result.name == "TRIMA_30"

        # 異なる期間でのテスト
        result2 = indicator.calculate(sample_data, 20)
        assert result2.name == "TRIMA_20"

        # 説明テスト
        assert "Triangular Moving Average" in indicator.get_description()

        # サポート期間テスト
        assert 30 in indicator.supported_periods
        assert 20 in indicator.supported_periods

    def test_factory_functions(self, sample_data):
        """ファクトリー関数のテスト"""
        # モメンタム系指標
        momentum_indicators = ["BOP", "PPO"]

        for indicator_type in momentum_indicators:
            indicator = get_momentum_indicator(indicator_type)
            assert indicator is not None
            assert indicator.indicator_type == indicator_type

            # 計算テスト
            if indicator_type == "BOP":
                result = indicator.calculate(sample_data, 1)
            elif indicator_type == "PPO":
                result = indicator.calculate(sample_data, 12, slow_period=26)

            assert isinstance(result, pd.Series)
            assert len(result) == len(sample_data)

        # トレンド系指標
        trend_indicators = ["MIDPOINT", "MIDPRICE", "TRIMA"]

        for indicator_type in trend_indicators:
            indicator = get_trend_indicator(indicator_type)
            assert indicator is not None
            assert indicator.indicator_type == indicator_type

            # 計算テスト
            if indicator_type == "TRIMA":
                result = indicator.calculate(sample_data, 30)
            else:
                result = indicator.calculate(sample_data, 14)

            assert isinstance(result, pd.Series)
            assert len(result) == len(sample_data)

    def test_error_handling(self, sample_data):
        """エラーハンドリングのテスト"""
        # BOP（open, high, low, closeが必要）
        bop = BOPIndicator()
        invalid_data = sample_data.drop(columns=["open"])
        with pytest.raises(ValueError, match="BOP計算には"):
            bop.calculate(invalid_data, 1)

        # MIDPRICE（high, lowが必要）
        midprice = MIDPRICEIndicator()
        invalid_data2 = sample_data.drop(columns=["high"])
        with pytest.raises(ValueError, match="MIDPRICE計算には"):
            midprice.calculate(invalid_data2, 14)

    def test_calculation_consistency(self, sample_data):
        """計算の一貫性テスト"""
        # 同じパラメータで複数回計算して結果が一致することを確認
        bop = BOPIndicator()
        result1 = bop.calculate(sample_data, 1)
        result2 = bop.calculate(sample_data, 1)

        pd.testing.assert_series_equal(result1, result2)

        # MIDPOINT計算の一貫性
        midpoint = MIDPOINTIndicator()
        result1 = midpoint.calculate(sample_data, 14)
        result2 = midpoint.calculate(sample_data, 14)

        pd.testing.assert_series_equal(result1, result2)

    def test_value_ranges(self, sample_data):
        """値の範囲テスト"""
        # BOP: -1 から 1 の範囲
        bop = BOPIndicator()
        bop_result = bop.calculate(sample_data, 1)
        valid_bop = bop_result.dropna()
        assert all(valid_bop >= -1)
        assert all(valid_bop <= 1)

        # MIDPOINT, MIDPRICE, TRIMA: 価格範囲内
        midpoint = MIDPOINTIndicator()
        midpoint_result = midpoint.calculate(sample_data, 14)
        valid_midpoint = midpoint_result.dropna()

        min_price = sample_data["close"].min()
        max_price = sample_data["close"].max()

        # 中点は価格範囲内にあるべき
        assert all(valid_midpoint >= min_price * 0.8)  # 多少の余裕を持たせる
        assert all(valid_midpoint <= max_price * 1.2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
