"""
VolumeグループのNaNインジケーター修正TDDテスト

KVO, PVO, VP
のNaN発生問題をpandas-ta Noneフォールバックでテスト
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from app.services.indicators.technical_indicators.volume import VolumeIndicators
from app.services.indicators.technical_indicators.momentum import MomentumIndicators

class TestVolumeNaNIndicatorsTDD:
    """VolumeグループNaNインジケーターのテストクラス"""

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

    def test_kvo_pandas_ta_none_handling(self, sample_ohlcv_data):
        """KVO pandas-ta Noneフォールバックテスト"""
        with patch('pandas_ta.kvo', return_value=None):
            kvo_result = VolumeIndicators.kvo(
                high=pd.Series(sample_ohlcv_data['high']),
                low=pd.Series(sample_ohlcv_data['low']),
                volume=pd.Series(sample_ohlcv_data['volume']),
                length=34,
                fast=5,
                slow=10
            )

            assert isinstance(kvo_result, tuple)
            assert len(kvo_result) == 3
            for series in kvo_result:
                assert isinstance(series, pd.Series)
                assert len(series) == len(sample_ohlcv_data['high'])
                assert series.isna().all() or len(series) == 0

    def test_pvo_pandas_ta_none_handling(self, sample_ohlcv_data):
        """PVO pandas-ta Noneフォールバックテスト"""
        with patch('pandas_ta.pvo', return_value=None):
            pvo_result = MomentumIndicators.pvo(
                volume=pd.Series(sample_ohlcv_data['volume']),
                slow=26,
                fast=12,
                signal=9
            )

            assert isinstance(pvo_result, tuple)
            assert len(pvo_result) == 3
            for series in pvo_result:
                assert isinstance(series, pd.Series)
                assert len(series) == len(sample_ohlcv_data['volume'])
                assert series.isna().all() or len(series) == 0

    def test_vp_pandas_ta_none_handling(self, sample_ohlcv_data):
        """VP pandas-ta Noneフォールバックテスト"""
        with patch('pandas_ta.vp', return_value=None):
            vp_result = VolumeIndicators.vp(
                close=pd.Series(sample_ohlcv_data['close']),
                volume=pd.Series(sample_ohlcv_data['volume']),
                width=10
            )

            assert isinstance(vp_result, tuple)
            assert len(vp_result) == 2
            for series in vp_result:
                assert isinstance(series, pd.Series)
                assert len(series) == len(sample_ohlcv_data['close'])
                assert series.isna().all() or len(series) == 0
