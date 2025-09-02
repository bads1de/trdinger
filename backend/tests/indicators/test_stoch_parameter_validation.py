"""
STOCH指標のパラメータ検証テスト
パラメータエラーの修正をテストおよび確認
"""
import pytest
import pandas as pd
import numpy as np


class TestSTOCHParameterValidation:
    """STOCH指標のパラメータ検証テスト"""

    def setup_sample_data(self, length=100):
        """テスト用サンプルデータを生成"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=length, freq='h')

        # OHLCデータを生成
        close = 100 + np.cumsum(np.random.randn(length) * 2)
        high = close * (1 + np.random.rand(length) * 0.02)
        low = close * (1 - np.random.rand(length) * 0.02)

        return {
            'high': pd.Series(high, index=dates),
            'low': pd.Series(low, index=dates),
            'close': pd.Series(close, index=dates)
        }

    @pytest.fixture
    def sample_data(self):
        """pytest fixture for sample data"""
        return self.setup_sample_data()

    def test_stoch_negative_k_parameter(self, sample_data):
        """負のk_periodパラメータでValueErrorが発生することを確認"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        # これらのテストは最初は失敗すべき（バリデーションがないため）
        with pytest.raises(ValueError, match="k must be positive"):
            MomentumIndicators.stoch(
                high=sample_data['high'],
                low=sample_data['low'],
                close=sample_data['close'],
                k=-5,  # 負値
                d=3,
                smooth_k=3
            )

    def test_stoch_negative_d_parameter(self, sample_data):
        """負のd_periodパラメータでValueErrorが発生することを確認"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        # これらのテストは最初は失敗すべき（バリデーションがないため）
        with pytest.raises(ValueError, match="d must be positive"):
            MomentumIndicators.stoch(
                high=sample_data['high'],
                low=sample_data['low'],
                close=sample_data['close'],
                k=14,
                d=-3,  # 負値
                smooth_k=3
            )

    def test_stoch_negative_smooth_k_parameter(self, sample_data):
        """負のsmooth_k_periodパラメータでValueErrorが発生することを確認"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        # これらのテストは最初は失敗すべき（バリデーションがないため）
        with pytest.raises(ValueError, match="smooth_k must be positive"):
            MomentumIndicators.stoch(
                high=sample_data['high'],
                low=sample_data['low'],
                close=sample_data['close'],
                k=14,
                d=3,
                smooth_k=-3  # 負値
            )

    def test_stoch_zero_k_parameter(self, sample_data):
        """ゼロのk_periodパラメータでValueErrorが発生することを確認"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        with pytest.raises(ValueError, match="k must be positive"):
            MomentumIndicators.stoch(
                high=sample_data['high'],
                low=sample_data['low'],
                close=sample_data['close'],
                k=0,  # ゼロ
                d=3,
                smooth_k=3
            )

    def test_stoch_zero_d_parameter(self, sample_data):
        """ゼロのd_periodパラメータでValueErrorが発生することを確認"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        with pytest.raises(ValueError, match="d must be positive"):
            MomentumIndicators.stoch(
                high=sample_data['high'],
                low=sample_data['low'],
                close=sample_data['close'],
                k=14,
                d=0,  # ゼロ
                smooth_k=3
            )

    def test_stoch_zero_smooth_k_parameter(self, sample_data):
        """ゼロのsmooth_k_periodパラメータでValueErrorが発生することを確認"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        with pytest.raises(ValueError, match="smooth_k must be positive"):
            MomentumIndicators.stoch(
                high=sample_data['high'],
                low=sample_data['low'],
                close=sample_data['close'],
                k=14,
                d=3,
                smooth_k=0  # ゼロ
            )

    def test_stoch_valid_parameters(self, sample_data):
        """有効なパラメータで正常に計算されることを確認"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        result = MomentumIndicators.stoch(
            high=sample_data['high'],
            low=sample_data['low'],
            close=sample_data['close'],
            k=14,
            d=3,
            smooth_k=3
        )

        # 結果の基本検証
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(series, pd.Series) for series in result)


if __name__ == "__main__":
    pytest.main([__file__])