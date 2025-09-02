"""
VolatilityグループのNaNインジケーター修正TDDテスト

BB, DONCHIAN, SUPERTREND, VORTEX
のNaN発生問題をpandas-ta Noneフォールバックでテスト
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from app.services.indicators.technical_indicators.volatility import VolatilityIndicators

class TestVolatilityNaNIndicatorsTDD:
    """VolatilityグループNaNインジケーターのテストクラス"""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """テスト用OHLCVデータを生成"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='h')

        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        high = close + np.random.rand(100) * 1
        low = close - np.random.rand(100) * 1
        volume = np.random.randint(1000, 10000, 100)

        return pd.DataFrame({
            'timestamp': dates,
            'open': close + np.random.randn(100) * 0.5,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    def test_bb_pandas_ta_none_handling(self, sample_ohlcv_data):
        """BB pandas-ta Noneフォールバックテスト"""
        from app.services.indicators.technical_indicators.volatility import VolatilityIndicators

        with patch('pandas_ta.bbands', return_value=None):
            bb_result = VolatilityIndicators.bbands(
                data=pd.Series(sample_ohlcv_data['close']),
                length=20,
                std=2.0
            )

            assert isinstance(bb_result, tuple)
            assert len(bb_result) == 3
            for series in bb_result:
                assert isinstance(series, pd.Series)
                assert len(series) == len(sample_ohlcv_data['close'])
                assert series.isna().all() or len(series) == 0

    def test_donchian_pandas_ta_none_handling(self, sample_ohlcv_data):
        """DONCHIAN pandas-ta Noneフォールバックテスト"""
        from app.services.indicators.technical_indicators.volatility import VolatilityIndicators

        with patch('pandas_ta.donchian', return_value=None):
            donchian_result = VolatilityIndicators.donchian(
                high=pd.Series(sample_ohlcv_data['high']),
                low=pd.Series(sample_ohlcv_data['low']),
                length=20
            )

            assert isinstance(donchian_result, tuple)
            assert len(donchian_result) == 3
            for series in donchian_result:
                assert isinstance(series, pd.Series)
                assert len(series) == len(sample_ohlcv_data['high'])
                assert series.isna().all() or len(series) == 0

    def test_supertrend_pandas_ta_none_handling(self, sample_ohlcv_data):
        """SUPERTREND pandas-ta Noneフォールバックテスト"""
        from app.services.indicators.technical_indicators.volatility import VolatilityIndicators

        with patch('pandas_ta.supertrend', return_value=None):
            supertrend_result = VolatilityIndicators.supertrend(
                high=pd.Series(sample_ohlcv_data['high']),
                low=pd.Series(sample_ohlcv_data['low']),
                close=pd.Series(sample_ohlcv_data['close']),
                length=10,
                multiplier=3.0
            )

            assert isinstance(supertrend_result, tuple)
            assert len(supertrend_result) == 3
            for series in supertrend_result:
                assert isinstance(series, pd.Series)
                assert len(series) == len(sample_ohlcv_data['close'])
                # supertrendのテストでは一部NaNでもOKだが、最後の2つ（direction以外）はNaNチェック
                if series.name != 'direction' and series.name != 'dir':  # directionは特別扱い
                    assert series.isna().all() or len(series) == 0

    def test_vortex_pandas_ta_none_handling(self, sample_ohlcv_data):
        """VORTEX pandas-ta Noneフォールバックテスト"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        with patch('pandas_ta.vortex', return_value=None):
            vortex_result = MomentumIndicators.vortex(
                high=pd.Series(sample_ohlcv_data['high']),
                low=pd.Series(sample_ohlcv_data['low']),
                close=pd.Series(sample_ohlcv_data['close'])
            )

            # VORTEXは通常2つのSeriesを返す（VI+ and VI-）
            assert isinstance(vortex_result, tuple)
            assert len(vortex_result) == 2
            for series in vortex_result:
                assert isinstance(series, pd.Series)
                assert len(series) == len(sample_ohlcv_data['close'])
                assert series.isna().all() or len(series) == 0
