"""
strategies/indicators.pyとIndicatorOrchestratorの統合テスト

TDDアプローチで、strategies/indicators.py内の関数がIndicatorOrchestratorを
利用するように変更することを検証するテスト
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from typing import Union, List

from app.core.strategies.indicators import SMA, EMA, RSI, BollingerBands, ATR
from app.core.services.indicators.indicator_orchestrator import TechnicalIndicatorService


class TestStrategiesIndicatorsOrchestratorIntegration:
    """strategies/indicators.pyとIndicatorOrchestratorの統合テストクラス"""

    @pytest.fixture
    def sample_price_data(self):
        """テスト用の価格データを生成"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        return pd.Series(prices, index=dates, name='close')

    @pytest.fixture
    def sample_ohlcv_data(self):
        """テスト用のOHLCVデータを生成"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        high_prices = close_prices + np.random.uniform(0, 2, 100)
        low_prices = close_prices - np.random.uniform(0, 2, 100)
        open_prices = close_prices + np.random.uniform(-1, 1, 100)
        volume = np.random.randint(1000, 10000, 100)
        
        return {
            'high': pd.Series(high_prices, index=dates),
            'low': pd.Series(low_prices, index=dates),
            'close': pd.Series(close_prices, index=dates)
        }

    @pytest.fixture
    def mock_orchestrator(self):
        """モックされたIndicatorOrchestratorを作成"""
        return Mock(spec=TechnicalIndicatorService)

    def test_sma_function_uses_orchestrator(self, sample_price_data, mock_orchestrator):
        """SMA関数がIndicatorOrchestratorを使用することを検証"""
        period = 20
        expected_result = pd.Series([100.0] * len(sample_price_data), 
                                  index=sample_price_data.index, name='SMA_20')
        
        # IndicatorOrchestratorのモック設定
        mock_orchestrator.calculate_indicator.return_value = expected_result
        
        with patch('app.core.strategies.indicators.TechnicalIndicatorService') as mock_service_class:
            mock_service_class.return_value = mock_orchestrator
            
            # SMA関数を呼び出し
            result = SMA(sample_price_data, period)
            
            # IndicatorOrchestratorが正しく初期化されたことを確認
            mock_service_class.assert_called_once()
            
            # calculate_indicatorが正しい引数で呼び出されたことを確認
            mock_orchestrator.calculate_indicator.assert_called_once()
            call_args = mock_orchestrator.calculate_indicator.call_args
            
            # 引数の検証（データ、指標タイプ、期間）
            assert call_args[1]['indicator_type'] == 'SMA'
            assert call_args[1]['period'] == period
            
            # 結果が期待通りであることを確認
            pd.testing.assert_series_equal(result, expected_result)

    def test_ema_function_uses_orchestrator(self, sample_price_data, mock_orchestrator):
        """EMA関数がIndicatorOrchestratorを使用することを検証"""
        period = 20
        expected_result = pd.Series([100.0] * len(sample_price_data), 
                                  index=sample_price_data.index, name='EMA_20')
        
        mock_orchestrator.calculate_indicator.return_value = expected_result
        
        with patch('app.core.strategies.indicators.TechnicalIndicatorService') as mock_service_class:
            mock_service_class.return_value = mock_orchestrator
            
            result = EMA(sample_price_data, period)
            
            mock_service_class.assert_called_once()
            mock_orchestrator.calculate_indicator.assert_called_once()
            call_args = mock_orchestrator.calculate_indicator.call_args
            
            assert call_args[1]['indicator_type'] == 'EMA'
            assert call_args[1]['period'] == period
            pd.testing.assert_series_equal(result, expected_result)

    def test_rsi_function_uses_orchestrator(self, sample_price_data, mock_orchestrator):
        """RSI関数がIndicatorOrchestratorを使用することを検証"""
        period = 14
        expected_result = pd.Series([50.0] * len(sample_price_data), 
                                  index=sample_price_data.index, name='RSI_14')
        
        mock_orchestrator.calculate_indicator.return_value = expected_result
        
        with patch('app.core.strategies.indicators.TechnicalIndicatorService') as mock_service_class:
            mock_service_class.return_value = mock_orchestrator
            
            result = RSI(sample_price_data, period)
            
            mock_service_class.assert_called_once()
            mock_orchestrator.calculate_indicator.assert_called_once()
            call_args = mock_orchestrator.calculate_indicator.call_args
            
            assert call_args[1]['indicator_type'] == 'RSI'
            assert call_args[1]['period'] == period
            pd.testing.assert_series_equal(result, expected_result)

    def test_atr_function_uses_orchestrator(self, sample_ohlcv_data, mock_orchestrator):
        """ATR関数がIndicatorOrchestratorを使用することを検証"""
        period = 14
        expected_result = pd.Series([1.0] * len(sample_ohlcv_data['close']), 
                                  index=sample_ohlcv_data['close'].index, name='ATR_14')
        
        mock_orchestrator.calculate_indicator.return_value = expected_result
        
        with patch('app.core.strategies.indicators.TechnicalIndicatorService') as mock_service_class:
            mock_service_class.return_value = mock_orchestrator
            
            result = ATR(
                sample_ohlcv_data['high'], 
                sample_ohlcv_data['low'], 
                sample_ohlcv_data['close'], 
                period
            )
            
            mock_service_class.assert_called_once()
            mock_orchestrator.calculate_indicator.assert_called_once()
            call_args = mock_orchestrator.calculate_indicator.call_args
            
            assert call_args[1]['indicator_type'] == 'ATR'
            assert call_args[1]['period'] == period
            pd.testing.assert_series_equal(result, expected_result)

    def test_bollinger_bands_function_uses_orchestrator(self, sample_price_data, mock_orchestrator):
        """BollingerBands関数がIndicatorOrchestratorを使用することを検証"""
        period = 20
        std_dev = 2.0
        
        # BollingerBandsは辞書を返すため、期待される結果を設定
        expected_bb_result = {
            'upper': pd.Series([105.0] * len(sample_price_data), index=sample_price_data.index),
            'middle': pd.Series([100.0] * len(sample_price_data), index=sample_price_data.index),
            'lower': pd.Series([95.0] * len(sample_price_data), index=sample_price_data.index)
        }
        
        mock_orchestrator.calculate_indicator.return_value = expected_bb_result
        
        with patch('app.core.strategies.indicators.TechnicalIndicatorService') as mock_service_class:
            mock_service_class.return_value = mock_orchestrator
            
            result = BollingerBands(sample_price_data, period, std_dev)
            
            mock_service_class.assert_called_once()
            mock_orchestrator.calculate_indicator.assert_called_once()
            call_args = mock_orchestrator.calculate_indicator.call_args
            
            assert call_args[1]['indicator_type'] == 'BB'
            assert call_args[1]['period'] == period
            assert call_args[1]['std_dev'] == std_dev
            
            # 結果がtupleで返されることを確認（既存APIとの互換性）
            assert isinstance(result, tuple)
            assert len(result) == 3
            upper, middle, lower = result
            pd.testing.assert_series_equal(upper, expected_bb_result['upper'])
            pd.testing.assert_series_equal(middle, expected_bb_result['middle'])
            pd.testing.assert_series_equal(lower, expected_bb_result['lower'])

    def test_function_signatures_unchanged(self):
        """関数のシグネチャが変更されていないことを検証（後方互換性）"""
        import inspect
        
        # SMA関数のシグネチャ検証
        sma_sig = inspect.signature(SMA)
        sma_params = list(sma_sig.parameters.keys())
        assert 'data' in sma_params
        assert 'period' in sma_params
        
        # EMA関数のシグネチャ検証
        ema_sig = inspect.signature(EMA)
        ema_params = list(ema_sig.parameters.keys())
        assert 'data' in ema_params
        assert 'period' in ema_params
        
        # RSI関数のシグネチャ検証
        rsi_sig = inspect.signature(RSI)
        rsi_params = list(rsi_sig.parameters.keys())
        assert 'data' in rsi_params
        assert 'period' in rsi_params
        
        # ATR関数のシグネチャ検証
        atr_sig = inspect.signature(ATR)
        atr_params = list(atr_sig.parameters.keys())
        assert 'high' in atr_params
        assert 'low' in atr_params
        assert 'close' in atr_params
        assert 'period' in atr_params

    def test_no_direct_talib_adapter_usage(self):
        """strategies/indicators.py内でTALibAdapterが直接使用されていないことを検証"""
        import inspect
        from app.core.strategies import indicators
        
        source = inspect.getsource(indicators)
        # TALibAdapterの直接使用がないことを確認
        assert 'TALibAdapter.' not in source, "strategies/indicators.py still uses TALibAdapter directly"

    def test_orchestrator_usage_in_strategies_indicators(self):
        """strategies/indicators.py内でTechnicalIndicatorServiceが使用されていることを検証"""
        import inspect
        from app.core.strategies import indicators
        
        source = inspect.getsource(indicators)
        # TechnicalIndicatorServiceの使用があることを確認
        assert 'TechnicalIndicatorService' in source, "strategies/indicators.py does not use TechnicalIndicatorService"

    def test_data_conversion_compatibility(self, mock_orchestrator):
        """データ変換の互換性を検証（List、np.arrayからpd.Seriesへの変換）"""
        # List形式のデータ
        list_data = [100.0, 101.0, 102.0, 101.5, 103.0]
        
        # numpy array形式のデータ
        array_data = np.array([100.0, 101.0, 102.0, 101.5, 103.0])
        
        expected_result = pd.Series([100.0] * 5, name='SMA_5')
        mock_orchestrator.calculate_indicator.return_value = expected_result
        
        with patch('app.core.strategies.indicators.TechnicalIndicatorService') as mock_service_class:
            mock_service_class.return_value = mock_orchestrator
            
            # List形式のデータでテスト
            result_list = SMA(list_data, 5)
            pd.testing.assert_series_equal(result_list, expected_result)
            
            # numpy array形式のデータでテスト
            result_array = SMA(array_data, 5)
            pd.testing.assert_series_equal(result_array, expected_result)
