"""
テクニカル指標のテスト
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtest_engine.indicators import TechnicalIndicators


class TestTechnicalIndicators:
    """テクニカル指標のテストクラス"""
    
    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータを生成"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)  # 再現性のため
        
        # 簡単なランダムウォークで価格データを生成
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        data = pd.DataFrame({
            'open': prices + np.random.randn(100) * 0.1,
            'high': prices + np.abs(np.random.randn(100) * 0.3),
            'low': prices - np.abs(np.random.randn(100) * 0.3),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        return data
    
    def test_sma_calculation(self, sample_data):
        """SMA計算のテスト"""
        period = 20
        sma = TechnicalIndicators.sma(sample_data['close'], period)
        
        # 結果の長さが正しいか
        assert len(sma) == len(sample_data)
        
        # 最初のperiod-1個はNaNであるべき
        assert np.isnan(sma[:period-1]).all()
        
        # period番目以降は値が入っているべき
        assert not np.isnan(sma[period-1:]).any()
        
        # 手動計算との比較（最初の有効な値）
        expected_first_sma = sample_data['close'][:period].mean()
        assert abs(sma[period-1] - expected_first_sma) < 1e-10
    
    def test_rsi_calculation(self, sample_data):
        """RSI計算のテスト"""
        period = 14
        rsi = TechnicalIndicators.rsi(sample_data['close'], period)
        
        # 結果の長さが正しいか
        assert len(rsi) == len(sample_data)
        
        # RSIは0-100の範囲内であるべき
        valid_rsi = rsi[~np.isnan(rsi)]
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_macd_calculation(self, sample_data):
        """MACD計算のテスト"""
        macd_line, macd_signal, macd_histogram = TechnicalIndicators.macd(sample_data['close'])
        
        # 結果の長さが正しいか
        assert len(macd_line) == len(sample_data)
        assert len(macd_signal) == len(sample_data)
        assert len(macd_histogram) == len(sample_data)
        
        # ヒストグラムはMACDライン - シグナルラインであるべき
        valid_indices = ~(np.isnan(macd_line) | np.isnan(macd_signal))
        np.testing.assert_array_almost_equal(
            macd_histogram[valid_indices],
            macd_line[valid_indices] - macd_signal[valid_indices]
        )
    
    def test_bollinger_bands_calculation(self, sample_data):
        """ボリンジャーバンド計算のテスト"""
        upper, middle, lower = TechnicalIndicators.bollinger_bands(sample_data['close'])
        
        # 結果の長さが正しいか
        assert len(upper) == len(sample_data)
        assert len(middle) == len(sample_data)
        assert len(lower) == len(sample_data)
        
        # 有効な値において、upper > middle > lower であるべき
        valid_indices = ~(np.isnan(upper) | np.isnan(middle) | np.isnan(lower))
        assert (upper[valid_indices] >= middle[valid_indices]).all()
        assert (middle[valid_indices] >= lower[valid_indices]).all()
    
    def test_calculate_indicator_sma(self, sample_data):
        """汎用指標計算メソッドのテスト（SMA）"""
        result = TechnicalIndicators.calculate_indicator(
            sample_data, 'SMA', {'period': 20}
        )
        
        # 直接計算との比較
        expected = TechnicalIndicators.sma(sample_data['close'], 20)
        np.testing.assert_array_equal(result, expected)
    
    def test_calculate_indicator_rsi(self, sample_data):
        """汎用指標計算メソッドのテスト（RSI）"""
        result = TechnicalIndicators.calculate_indicator(
            sample_data, 'RSI', {'period': 14}
        )
        
        # 直接計算との比較
        expected = TechnicalIndicators.rsi(sample_data['close'], 14)
        np.testing.assert_array_equal(result, expected)
    
    def test_calculate_indicator_macd(self, sample_data):
        """汎用指標計算メソッドのテスト（MACD）"""
        result = TechnicalIndicators.calculate_indicator(
            sample_data, 'MACD', {}
        )
        
        # 結果が辞書であることを確認
        assert isinstance(result, dict)
        assert 'macd' in result
        assert 'signal' in result
        assert 'histogram' in result
    
    def test_unsupported_indicator(self, sample_data):
        """サポートされていない指標のテスト"""
        with pytest.raises(ValueError, match="Unsupported indicator"):
            TechnicalIndicators.calculate_indicator(
                sample_data, 'UNKNOWN_INDICATOR', {}
            )


if __name__ == "__main__":
    pytest.main([__file__])
