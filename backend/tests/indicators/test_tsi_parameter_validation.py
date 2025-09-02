"""
TSI指標のパラメータ検証テスト
パラメータエラーの修正をテストおよび確認
"""
import pytest
import pandas as pd
import numpy as np


class TestTSIParameterValidation:
    """TSI指標のパラメータ検証テスト"""

    def setup_sample_data(self, length=100):
        """テスト用サンプルデータを生成"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=length, freq='h')

        # OHLCデータを生成
        close = 100 + np.cumsum(np.random.randn(length) * 2)

        return {
            'close': pd.Series(close, index=dates)
        }

    @pytest.fixture
    def sample_data(self):
        """pytest fixture for sample data"""
        return self.setup_sample_data()

    def test_tsi_negative_fast_parameter(self, sample_data):
        """負のfast_periodパラメータでValueErrorが発生することを確認"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        # これらのテストは最初は失敗すべき（バリデーションがないため）
        with pytest.raises(ValueError, match="fast must be positive"):
            MomentumIndicators.tsi(
                data=sample_data['close'],
                fast=-5,  # 負値
                slow=25
            )

    def test_tsi_negative_slow_parameter(self, sample_data):
        """負のslow_periodパラメータでValueErrorが発生することを確認"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        # これらのテストは最初は失敗すべき（バリデーションがないため）
        with pytest.raises(ValueError, match="slow must be positive"):
            MomentumIndicators.tsi(
                data=sample_data['close'],
                fast=13,
                slow=-10  # 負値
            )

    def test_tsi_zero_fast_parameter(self, sample_data):
        """ゼロのfast_periodパラメータでValueErrorが発生することを確認"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        with pytest.raises(ValueError, match="fast must be positive"):
            MomentumIndicators.tsi(
                data=sample_data['close'],
                fast=0,  # ゼロ
                slow=25
            )

    def test_tsi_zero_slow_parameter(self, sample_data):
        """ゼロのslow_periodパラメータでValueErrorが発生することを確認"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        with pytest.raises(ValueError, match="slow must be positive"):
            MomentumIndicators.tsi(
                data=sample_data['close'],
                fast=13,
                slow=0  # ゼロ
            )

    def test_tsi_fast_greater_than_slow(self, sample_data):
        """fastがslowより大きい場合のテスト（オプション）"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        # TSIの場合はfast > slowでも数学的に問題ないかもしれない
        # このテストはケースバイケースで調整
        result = MomentumIndicators.tsi(
            data=sample_data['close'],
            fast=25,
            slow=13  # fast > slow
        )
        # 結果があればOK（pandas-taが適切に処理してくれる）
        assert result is not None

    def test_tsi_valid_parameters(self, sample_data):
        """有効なパラメータで正常に計算されることを確認"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        result = MomentumIndicators.tsi(
            data=sample_data['close'],
            fast=13,
            slow=25
        )

        # 結果の基本検証
        assert result is not None
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data['close'])


if __name__ == "__main__":
    pytest.main([__file__])