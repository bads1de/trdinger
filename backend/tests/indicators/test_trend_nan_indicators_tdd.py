"""
TrendグループのNaNインジケーター修正TDDテスト

HWC, ICHIMOKU
のNaN発生問題をpandas-ta Noneフォールバックでテスト
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from app.services.indicators.technical_indicators.trend import TrendIndicators
from app.services.indicators.technical_indicators.volatility import VolatilityIndicators

class TestTrendNaNIndicatorsTDD:
    """TrendグループNaNインジケーターのテストクラス"""

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

    def test_hwc_pandas_ta_none_handling(self, sample_ohlcv_data):
        """HWC pandas-ta Noneフォールバックテスト"""
        with patch('pandas_ta.hwc', return_value=None):
            hwc_result = VolatilityIndicators.hwc(
                close=pd.Series(sample_ohlcv_data['close']),
                na=0.2,
                nb=0.1,
                nc=3.0,
                nd=0.3,
                scalar=2.0
            )

            assert isinstance(hwc_result, tuple)
            assert len(hwc_result) == 3
            for series in hwc_result:
                assert isinstance(series, pd.Series)
                assert len(series) == len(sample_ohlcv_data['close'])
                assert series.isna().all() or len(series) == 0

    def test_ichimoku_pandas_ta_none_handling(self, sample_ohlcv_data):
        """ICHIMOKU pandas-ta Noneフォールバックテスト"""
        with patch('pandas_ta.ichimoku', return_value=None):
            ichimoku_result = TrendIndicators.ichimoku_cloud(
                high=pd.Series(sample_ohlcv_data['high']),
                low=pd.Series(sample_ohlcv_data['low']),
                close=pd.Series(sample_ohlcv_data['close']),
                conversion_line_period=9,
                base_line_period=26,
                span_b_period=52,
                lagging_span_period=26
            )

            # ICHIMOKUは通常5つのSeriesを返す（converting line, base line, leadingA, leadingB, lagging span）
            assert isinstance(ichimoku_result, tuple)
            assert len(ichimoku_result) == 5
            for series in ichimoku_result:
                assert isinstance(series, pd.Series)
                assert len(series) == len(sample_ohlcv_data['close'])
                assert series.isna().all() or len(series) == 0
