"""
SUPERTRENDインジケーターのfactorパラメータ追加テスト

TDD方式により、まずは失敗することを確認し、その後修正を実装
"""

import pytest
import pandas as pd
import numpy as np

from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService


class TestSuperttrendFactorParameter:
    """SUPERTRENDのfactorパラメータのテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用データ作成"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=50, freq='h')

        base_price = 50000
        price_changes = np.random.normal(0, 0.01, 50)

        close_prices = [base_price]
        for change in price_changes[1:]:
            new_price = close_prices[-1] * (1 + change)
            close_prices.append(max(1, new_price))

        high_prices = [price * (1 + abs(np.random.normal(0, 0.005))) for price in close_prices]
        low_prices = [price * (1 - abs(np.random.normal(0, 0.005))) for price in close_prices]
        open_prices = [close_prices[i-1] * (1 + np.random.normal(0, 0.005)) for i in range(len(close_prices))]
        open_prices[0] = base_price * (1 + np.random.normal(0, 0.005))

        return pd.DataFrame({
            'timestamp': dates,
            'open': open_prices,
            'close': close_prices,
            'high': high_prices,
            'low': low_prices,
        })

    def test_superttrend_with_factor_parameter_should_succeed(self, sample_data):
        """factorパラメータでSUPERTRENDが成功することを確認（TDD: Red -> Green）"""
        service = TechnicalIndicatorService()

        # factorパラメータを使用するテスト
        params = {
            'length': 10,
            'factor': 2.0  # multiplierの別名としてfactorを使用
        }

        result = service.calculate_indicator(sample_data, 'SUPERTREND', params)

        # 結果は3つの配列からなるタプル（lower, upper, direction）
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 3

        # 各配列の検証
        for arr in result:
            if isinstance(arr, pd.Series):
                arr = arr.values  # Seriesをnumpy配列に変換
            assert isinstance(arr, np.ndarray)
            assert len(arr) == len(sample_data)
            # 最初の10個がNaNでも、後半に有効値があればOK (length=10の場合)
            valid_values = len(arr) - np.sum(np.isnan(arr))
            assert valid_values > len(sample_data) * 0.5  # 少なくとも50%が有効値

    def test_superttrend_different_factor_values(self, sample_data):
        """異なるfactor値での動作確認"""
        service = TechnicalIndicatorService()

        factors = [1.0, 2.0, 3.0, 5.0]

        for factor in factors:
            params = {
                'length': 10,
                'factor': factor
            }

            result = service.calculate_indicator(sample_data, 'SUPERTREND', params)

            assert result is not None
            assert isinstance(result, tuple)
            assert len(result) == 3

    def test_superttrend_factor_vs_multiplier_equivalence(self, sample_data):
        """factorとmultiplierのパラメータが等価であることを確認"""
        service = TechnicalIndicatorService()

        factor_result = service.calculate_indicator(
            sample_data, 'SUPERTREND',
            {'length': 10, 'factor': 3.0}
        )

        multiplier_result = service.calculate_indicator(
            sample_data, 'SUPERTREND',
            {'length': 10, 'multiplier': 3.0}
        )

        # 結果が同じであることを確認
        assert factor_result is not None
        assert multiplier_result is not None

        assert np.allclose(factor_result[0], multiplier_result[0], equal_nan=True)
        assert np.allclose(factor_result[1], multiplier_result[1], equal_nan=True)
        assert np.allclose(factor_result[2], multiplier_result[2], equal_nan=True)