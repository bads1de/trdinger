"""
新しいIndicatorCalculator（オートストラテジー最適化版）のテスト

numpy配列ベースの高速指標計算をテストします。
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from app.core.services.auto_strategy.factories.indicator_calculator import IndicatorCalculator
from app.core.services.indicators.utils import TALibError


class TestIndicatorCalculatorOptimized:
    """新しいIndicatorCalculator（オートストラテジー最適化版）のテスト"""

    @pytest.fixture
    def calculator(self):
        """IndicatorCalculatorインスタンス"""
        return IndicatorCalculator()

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータ"""
        np.random.seed(42)
        size = 100
        
        # 価格データ生成
        close_prices = 100 + np.cumsum(np.random.randn(size) * 0.5)
        high_prices = close_prices + np.random.rand(size) * 2
        low_prices = close_prices - np.random.rand(size) * 2
        volume = np.random.randint(1000, 10000, size)
        
        return {
            'close': close_prices,
            'high': high_prices,
            'low': low_prices,
            'volume': volume
        }

    def test_sma_calculation(self, calculator, sample_data):
        """SMA計算のテスト"""
        result, indicator_name = calculator.calculate_indicator(
            indicator_type='SMA',
            parameters={'period': 20},
            close_data=sample_data['close']
        )
        
        assert result is not None
        assert indicator_name == 'SMA'
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data['close'])
        
        # 最初の19個はNaNであることを確認
        assert np.isnan(result[:19]).all()
        # 20個目以降は有効な値であることを確認
        assert not np.isnan(result[19:]).any()

    def test_ema_calculation(self, calculator, sample_data):
        """EMA計算のテスト"""
        result, indicator_name = calculator.calculate_indicator(
            indicator_type='EMA',
            parameters={'period': 14},
            close_data=sample_data['close']
        )
        
        assert result is not None
        assert indicator_name == 'EMA'
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data['close'])

    def test_rsi_calculation(self, calculator, sample_data):
        """RSI計算のテスト"""
        result, indicator_name = calculator.calculate_indicator(
            indicator_type='RSI',
            parameters={'period': 14},
            close_data=sample_data['close']
        )
        
        assert result is not None
        assert indicator_name == 'RSI'
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data['close'])
        
        # RSIは0-100の範囲内であることを確認
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= 0)
        assert np.all(valid_values <= 100)

    def test_macd_calculation(self, calculator, sample_data):
        """MACD計算のテスト"""
        result, indicator_name = calculator.calculate_indicator(
            indicator_type='MACD',
            parameters={'fast': 12, 'slow': 26, 'signal': 9},
            close_data=sample_data['close']
        )
        
        assert result is not None
        assert indicator_name == 'MACD'
        assert isinstance(result, tuple)
        assert len(result) == 3  # MACD, Signal, Histogram
        
        macd, signal, histogram = result
        assert isinstance(macd, np.ndarray)
        assert isinstance(signal, np.ndarray)
        assert isinstance(histogram, np.ndarray)

    def test_atr_calculation(self, calculator, sample_data):
        """ATR計算のテスト（OHLC必要）"""
        result, indicator_name = calculator.calculate_indicator(
            indicator_type='ATR',
            parameters={'period': 14},
            close_data=sample_data['close'],
            high_data=sample_data['high'],
            low_data=sample_data['low']
        )
        
        assert result is not None
        assert indicator_name == 'ATR'
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data['close'])

    def test_bollinger_bands_calculation(self, calculator, sample_data):
        """ボリンジャーバンド計算のテスト"""
        result, indicator_name = calculator.calculate_indicator(
            indicator_type='BBANDS',
            parameters={'period': 20, 'std_dev': 2.0},
            close_data=sample_data['close']
        )
        
        assert result is not None
        assert indicator_name == 'BBANDS'
        assert isinstance(result, tuple)
        assert len(result) == 3  # Upper, Middle, Lower
        
        upper, middle, lower = result
        assert isinstance(upper, np.ndarray)
        assert isinstance(middle, np.ndarray)
        assert isinstance(lower, np.ndarray)

    def test_pandas_series_input(self, calculator, sample_data):
        """pandas Series入力のテスト"""
        close_series = pd.Series(sample_data['close'])
        
        result, indicator_name = calculator.calculate_indicator(
            indicator_type='SMA',
            parameters={'period': 20},
            close_data=close_series
        )
        
        assert result is not None
        assert indicator_name == 'SMA'
        assert isinstance(result, np.ndarray)

    def test_missing_required_data(self, calculator, sample_data):
        """必要なデータが不足している場合のテスト"""
        result, indicator_name = calculator.calculate_indicator(
            indicator_type='ATR',
            parameters={'period': 14},
            close_data=sample_data['close']
            # high_data, low_dataが不足
        )
        
        assert result is None
        assert indicator_name is None

    def test_unsupported_indicator(self, calculator, sample_data):
        """サポートされていない指標のテスト"""
        result, indicator_name = calculator.calculate_indicator(
            indicator_type='UNKNOWN_INDICATOR',
            parameters={'period': 14},
            close_data=sample_data['close']
        )
        
        assert result is None
        assert indicator_name is None

    def test_invalid_parameters(self, calculator, sample_data):
        """無効なパラメータのテスト"""
        result, indicator_name = calculator.calculate_indicator(
            indicator_type='SMA',
            parameters={'period': -1},  # 無効な期間
            close_data=sample_data['close']
        )
        
        assert result is None
        assert indicator_name is None

    def test_insufficient_data(self, calculator):
        """データ不足のテスト"""
        short_data = np.array([100, 101, 102])  # 3個のデータのみ
        
        result, indicator_name = calculator.calculate_indicator(
            indicator_type='SMA',
            parameters={'period': 20},  # 20期間必要
            close_data=short_data
        )
        
        assert result is None
        assert indicator_name is None

    def test_stochastic_calculation(self, calculator, sample_data):
        """ストキャスティクス計算のテスト"""
        result, indicator_name = calculator.calculate_indicator(
            indicator_type='STOCH',
            parameters={'fastk_period': 5, 'slowk_period': 3, 'slowd_period': 3},
            close_data=sample_data['close'],
            high_data=sample_data['high'],
            low_data=sample_data['low']
        )
        
        assert result is not None
        assert indicator_name == 'STOCH'
        assert isinstance(result, tuple)
        assert len(result) == 2  # %K, %D
        
        k_percent, d_percent = result
        assert isinstance(k_percent, np.ndarray)
        assert isinstance(d_percent, np.ndarray)

    def test_performance_optimization(self, calculator, sample_data):
        """パフォーマンス最適化の確認"""
        import time
        
        # 大きなデータセットでテスト
        large_data = np.random.randn(10000) + 100
        
        start_time = time.time()
        result, _ = calculator.calculate_indicator(
            indicator_type='SMA',
            parameters={'period': 20},
            close_data=large_data
        )
        end_time = time.time()
        
        # 計算が完了することを確認
        assert result is not None
        assert isinstance(result, np.ndarray)
        
        # 計算時間が合理的であることを確認（1秒以内）
        calculation_time = end_time - start_time
        assert calculation_time < 1.0, f"計算時間が長すぎます: {calculation_time}秒"

    def test_memory_efficiency(self, calculator, sample_data):
        """メモリ効率の確認"""
        # numpy配列が直接返されることを確認
        result, _ = calculator.calculate_indicator(
            indicator_type='SMA',
            parameters={'period': 20},
            close_data=sample_data['close']
        )
        
        assert isinstance(result, np.ndarray)
        # pandas Seriesではないことを確認
        assert not isinstance(result, pd.Series)
