"""
MomentumグループのNaNインジケーター修正TDDテスト

AROON, PPO, MACDEXT, MACDFIX, RVGI, SMI, STOCH, STOCHF, STOCHRSI
のNaN発生問題をpandas-ta Noneフォールバックでテスト
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from app.services.indicators.technical_indicators.momentum import MomentumIndicators

class TestMomentumNaNIndicatorsTDD:
    """MomentumグループNaNインジケーターのテストクラス"""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """テスト用OHLCVデータを生成"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='h')

        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        high = close + np.random.rand(100) * 2
        low = close - np.random.rand(100) * 2
        volume = np.random.randint(1000, 10000, 100)

        return pd.DataFrame({
            'timestamp': dates,
            'open': close + np.random.randn(100) * 1,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    def test_aroon_pandas_ta_none_handling(self, sample_ohlcv_data):
        """AROON pandas-ta Noneフォールバックテスト"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        with patch('pandas_ta.aroon', return_value=None):
            aroon_up, aroon_down = MomentumIndicators.aroon(
                high=pd.Series(sample_ohlcv_data['high']),
                low=pd.Series(sample_ohlcv_data['low']),
                period=14
            )

            assert isinstance(aroon_up, pd.Series)
            assert isinstance(aroon_down, pd.Series)
            assert len(aroon_up) == len(sample_ohlcv_data['high'])
            assert len(aroon_down) == len(sample_ohlcv_data['high'])
            assert aroon_up.isna().all() or len(aroon_up) == 0
            assert aroon_down.isna().all() or len(aroon_down) == 0

    def test_ppo_pandas_ta_none_handling(self, sample_ohlcv_data):
        """PPO pandas-ta Noneフォールバックテスト"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        with patch('pandas_ta.ppo', return_value=None):
            ppo_result = MomentumIndicators.ppo(
                data=pd.Series(sample_ohlcv_data['close']),
                fast=12,
                slow=26,
                signal=9
            )

            # PPOは通常tupleを返す（ppo, signal, histogram）
            assert isinstance(ppo_result, tuple)
            assert len(ppo_result) == 3
            for series in ppo_result:
                assert isinstance(series, pd.Series)
                assert len(series) == len(sample_ohlcv_data['close'])
                assert series.isna().all() or len(series) == 0

    def test_macdext_pandas_ta_none_handling(self, sample_ohlcv_data):
        """MACDEXT pandas-ta Noneフォールバックテスト"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        with patch('pandas_ta.macdext', return_value=None):
            macdext_result = MomentumIndicators.macdext(
                data=pd.Series(sample_ohlcv_data['close'])
            )

            assert isinstance(macdext_result, tuple)
            assert len(macdext_result) == 3
            for series in macdext_result:
                assert isinstance(series, pd.Series)
                assert len(series) == len(sample_ohlcv_data['close'])
                assert series.isna().all() or len(series) == 0

    def test_macdfix_pandas_ta_none_handling(self, sample_ohlcv_data):
        """MACDFIX pandas-ta Noneフォールバックテスト"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        with patch('pandas_ta.macdfix', return_value=None):
            macdfix_result = MomentumIndicators.macdfix(
                data=pd.Series(sample_ohlcv_data['close'])
            )

            assert isinstance(macdfix_result, tuple)
            assert len(macdfix_result) == 3
            for series in macdfix_result:
                assert isinstance(series, pd.Series)
                assert len(series) == len(sample_ohlcv_data['close'])
                assert series.isna().all() or len(series) == 0

    def test_rvgi_pandas_ta_none_handling(self, sample_ohlcv_data):
        """RVGI pandas-ta Noneフォールバックテスト"""
        with patch('pandas_ta.rvgi', return_value=None):
            # RVGIの実装クラスをインポート
            from app.services.indicators.technical_indicators.momentum import MomentumIndicators

            rvgi_result = MomentumIndicators.rvgi(
                open=pd.Series(sample_ohlcv_data['open']),
                high=pd.Series(sample_ohlcv_data['high']),
                low=pd.Series(sample_ohlcv_data['low']),
                close=pd.Series(sample_ohlcv_data['close'])
            )

            assert isinstance(rvgi_result, tuple)
            assert len(rvgi_result) == 2
            for series in rvgi_result:
                assert isinstance(series, pd.Series)
                assert len(series) == len(sample_ohlcv_data['close'])
                assert series.isna().all() or len(series) == 0

    def test_smi_pandas_ta_none_handling(self, sample_ohlcv_data):
        """SMI pandas-ta Noneフォールバックテスト"""
        with patch('pandas_ta.smi', return_value=None):
            from app.services.indicators.technical_indicators.momentum import MomentumIndicators

            smi_result = MomentumIndicators.smi(
                high=pd.Series(sample_ohlcv_data['high']),
                low=pd.Series(sample_ohlcv_data['low']),
                close=pd.Series(sample_ohlcv_data['close'])
            )

            assert isinstance(smi_result, pd.Series)
            assert len(smi_result) == len(sample_ohlcv_data['close'])
            assert smi_result.isna().all() or len(smi_result) == 0

    def test_stoch_pandas_ta_none_handling(self, sample_ohlcv_data):
        """STOCH pandas-ta Noneフォールバックテスト"""
        with patch('pandas_ta.stoch', return_value=None):
            from app.services.indicators.technical_indicators.momentum import MomentumIndicators

            stoch_result = MomentumIndicators.stoch(
                high=pd.Series(sample_ohlcv_data['high']),
                low=pd.Series(sample_ohlcv_data['low']),
                close=pd.Series(sample_ohlcv_data['close'])
            )

            assert isinstance(stoch_result, tuple)
            assert len(stoch_result) == 2
            for series in stoch_result:
                assert isinstance(series, pd.Series)
                assert len(series) == len(sample_ohlcv_data['close'])
                assert series.isna().all() or len(series) == 0

    def test_stochf_pandas_ta_none_handling(self, sample_ohlcv_data):
        """STOCHF pandas-ta Noneフォールバックテスト"""
        with patch('pandas_ta.stochf', return_value=None):
            from app.services.indicators.technical_indicators.momentum import MomentumIndicators

            stochf_result = MomentumIndicators.stochf(
                high=pd.Series(sample_ohlcv_data['high']),
                low=pd.Series(sample_ohlcv_data['low']),
                close=pd.Series(sample_ohlcv_data['close'])
            )

            assert isinstance(stochf_result, tuple)
            assert len(stochf_result) == 2
            for series in stochf_result:
                assert isinstance(series, pd.Series)
                assert len(series) == len(sample_ohlcv_data['close'])
                assert series.isna().all() or len(series) == 0

    def test_stochrsi_pandas_ta_none_handling(self, sample_ohlcv_data):
        """STOCHRSI pandas-ta Noneフォールバックテスト"""
        with patch('pandas_ta.stochrsi', return_value=None):
            from app.services.indicators.technical_indicators.momentum import MomentumIndicators

            stochrsi_result = MomentumIndicators.stochrsi(
                data=pd.Series(sample_ohlcv_data['close'])
            )

            assert isinstance(stochrsi_result, tuple)
            assert len(stochrsi_result) == 3
            for series in stochrsi_result:
                assert isinstance(series, pd.Series)
                assert len(series) == len(sample_ohlcv_data['close'])
                assert series.isna().all() or len(series) == 0
