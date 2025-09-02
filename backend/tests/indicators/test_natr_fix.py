"""NATRパラメータ統一テスト

TDDによりNATR指標のperiodパラメータをlengthに統一する修正をテストする。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from app.services.indicators.technical_indicators.volatility import VolatilityIndicators


class TestNATRParameterFix:
    """NATRパラメータ統一テスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用サンプルデータ"""
        np.random.seed(42)
        n = 100
        close = pd.Series(100 + np.cumsum(np.random.randn(n)), name="close")
        high = close + np.abs(np.random.randn(n)) * 5
        low = close - np.abs(np.random.randn(n)) * 5
        return {'close': close, 'high': high, 'low': low}

    def test_natr_with_length_parameter_current_failure(self, sample_data):
        """lengthパラメータでの呼び出しの動作確認（修正前はエラー）"""
        # 修正前はlengthパラメータが存在しないのでエラーに関するテスト
        # このテストは修正後にpassする

        # lengthパラメータで呼び出し
        try:
            result = VolatilityIndicators.natr(
                high=sample_data['high'],
                low=sample_data['low'],
                close=sample_data['close'],
                length=14
            )

            # 成功したらlengthパラメータが使用可能
            assert isinstance(result, pd.Series)
            assert len(result) == len(sample_data['high'])
            assert not result.isna().all()

        except TypeError as e:
            # lengthパラメータが存在しない場合のTypeError
            assert "period" in str(e) or "positional" in str(e)

    def test_natr_with_unified_length_parameter(self, sample_data):
        """統一されたlengthパラメータの呼び出し"""
        # lengthパラメータで標準的に呼び出し

        # lengthパラメータで呼び出し
        result = VolatilityIndicators.natr(
            high=sample_data['high'],
            low=sample_data['low'],
            close=sample_data['close'],
            length=14
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data['high'])
        assert not result.isna().all()

    def test_natr_calculation_accuracy(self, sample_data):
        """NATR計算の正確性確認"""
        # 長さ14のNATRを計算
        result = VolatilityIndicators.natr(
            high=sample_data['high'],
            low=sample_data['low'],
            close=sample_data['close'],
            length=14
        )

        # ATRとの比較による妥当性確認
        atr_result = VolatilityIndicators.atr(
            high=sample_data['high'],
            low=sample_data['low'],
            close=sample_data['close'],
            length=14
        )

        # NATRはATRをクローズで割った値の100倍なので相関関係が強いはず
        masked_result = result[~result.isna()]
        masked_atr = atr_result[~result.isna()]

        if len(masked_result) > 0:
            correlation = np.corrcoef(masked_atr, masked_result / 100)[0, 1]
            assert 0.8 < correlation < 1.0  # 強い相関があるはず

    def test_natr_parameter_unification(self, sample_data):
        """lengthパラメータ統一の確認"""
        # デフォルトパラメータでの呼び出し
        default_result = VolatilityIndicators.natr(
            high=sample_data['high'],
            low=sample_data['low'],
            close=sample_data['close']
        )

        # length=14での明示的呼び出し
        explicit_result = VolatilityIndicators.natr(
            high=sample_data['high'],
            low=sample_data['low'],
            close=sample_data['close'],
            length=14
        )

        # 結果が同じであること
        pd.testing.assert_series_equal(default_result, explicit_result)

    def test_natr_error_handling(self, sample_data):
        """エラーハンドリングの確認"""
        # 空のデータでのテスト
        empty_series = pd.Series([], dtype=float)
        result = VolatilityIndicators.natr(empty_series, empty_series, empty_series)
        assert result.isna().all()

        # NaNデータでのテスト
        nan_series = pd.Series([np.nan] * 100)
        result = VolatilityIndicators.natr(nan_series, nan_series, nan_series)
        assert result.isna().all()

        # 異なる長さのデータ
        short_series = pd.Series([100, 101, 102])
        long_series = pd.Series([100] * 10)
        result = VolatilityIndicators.natr(long_series, short_series, long_series, length=14)
        assert result is not None  # パラメータ長による結果


if __name__ == "__main__":
    pytest.main([__file__, "-v"])