"""
テストケース: DONCHIAN指標のパラメータ名統一（period → length）
"""

import pytest
import pandas as pd
import numpy as np
from app.services.indicators.technical_indicators.volatility import VolatilityIndicators


class TestDonchianParameterFix:
    """DONCHIAN指標のパラメータ名統一テスト"""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """サンプルOHLCVデータ"""
        np.random.seed(42)
        n = 30

        # 基礎データ生成
        base_price = 100
        high_pct = np.random.normal(0.01, 0.02, n)
        low_pct = np.random.normal(-0.01, 0.02, n)
        close_pct = np.random.normal(0, 0.015, n)

        high = pd.Series([base_price * np.prod(high_pct[:i+1] + 1) for i in range(n)])
        low = pd.Series([max(high.iloc[i] * 0.98, base_price - abs(low_pct[i]) * 10) for i in range(n)])
        close = pd.Series([low.iloc[i] + (high.iloc[i] - low.iloc[i]) * 0.5 for i in range(n)])

        # 価格の論理的一貫性を確保
        high = pd.Series([max(h, l, c) for h, l, c in zip(high, low, close)])
        low = pd.Series([min(h, l, c) for h, l, c in zip(high, low, close)])

        return {
            'high': high,
            'low': low,
            'close': close
        }

    def test_donchian_length_parameter_success(self, sample_ohlcv_data):
        """lengthパラメータを使用した正常ケース"""
        high = pd.Series(sample_ohlcv_data['high'])
        low = pd.Series(sample_ohlcv_data['low'])

        # 新しいlengthパラメータでテスト（期待される成功ケース）
        result = VolatilityIndicators.donchian(high=high, low=low, length=20)

        # タプル形式（upper, middle, lower）を確認
        assert isinstance(result, tuple)
        assert len(result) == 3

        upper, middle, lower = result
        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)

        # データ長が一致することを確認
        expected_length = len(high)
        assert len(upper) == expected_length
        assert len(middle) == expected_length
        assert len(lower) == expected_length

        # Middle bandはupperとlowerの平均であることを確認
        expected_middle = (upper + lower) / 2
        pd.testing.assert_series_equal(middle, expected_middle, atol=0.001)

    def test_donchian_current_period_parameter_should_fail(self, sample_ohlcv_data):
        """現在のperiodパラメータを使用したエラー再現テスト"""
        high = pd.Series(sample_ohlcv_data['high'])
        low = pd.Series(sample_ohlcv_data['low'])

        # 現在のperiodパラメータを使用してエラーを再現
        with pytest.raises(TypeError, match="donchian.*unexpected keyword argument.*period"):
            VolatilityIndicators.donchian(high=high, low=low, period=20)

    def test_donchian_default_length_parameter(self, sample_ohlcv_data):
        """デフォルトのlengthパラメータ値でのテスト"""
        high = pd.Series(sample_ohlcv_data['high'])
        low = pd.Series(sample_ohlcv_data['low'])

        # デフォルトパラメータでのテスト
        result_default = VolatilityIndicators.donchian(high=high, low=low)
        result_explicit = VolatilityIndicators.donchian(high=high, low=low, length=20)

        # デフォルト値と明示的値が同じ結果を返すことを確認
        upper_default, middle_default, lower_default = result_default
        upper_explicit, middle_explicit, lower_explicit = result_explicit

        pd.testing.assert_series_equal(upper_default, upper_explicit)
        pd.testing.assert_series_equal(middle_default, middle_explicit)
        pd.testing.assert_series_equal(lower_default, lower_explicit)

    def test_donchian_different_length_values(self, sample_ohlcv_data):
        """異なるlength値での結果確認"""
        high = pd.Series(sample_ohlcv_data['high'])
        low = pd.Series(sample_ohlcv_data['low'])

        # 異なるlength値でのテスト
        result_10 = VolatilityIndicators.donchian(high=high, low=low, length=10)
        result_30 = VolatilityIndicators.donchian(high=high, low=low, length=30)

        # 異なるlengthは異なる結果を返すことを基本確認（NaNの数が異なる）
        nan_count_10 = result_10[0].isna().sum()
        nan_count_30 = result_30[0].isna().sum()

        # length=30の方がNaNが多いはず（データ長が30しかない場合）
        assert nan_count_30 >= nan_count_10

    def test_donchian_dataframe_return_structure(self, sample_ohlcv_data):
        """戻り値が正しい構造を持つことを確認"""
        high = pd.Series(sample_ohlcv_data['high'])
        low = pd.Series(sample_ohlcv_data['low'])

        result = VolatilityIndicators.donchian(high=high, low=low, length=20)
        upper, middle, lower = result

        # インデックスが一致することを確認
        pd.testing.assert_index_equal(upper.index, high.index)
        pd.testing.assert_index_equal(middle.index, high.index)
        pd.testing.assert_index_equal(lower.index, high.index)

        # upper >= middle >= lower の関係が成立することを確認（NaN以外）
        valid_mask = ~(upper.isna() | middle.isna() | lower.isna())
        if valid_mask.any():
            assert (upper[valid_mask] >= middle[valid_mask]).all()
            assert (middle[valid_mask] >= lower[valid_mask]).all()