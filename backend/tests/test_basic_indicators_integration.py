"""
基本指標移行後の統合テスト

移行したSMA, EMA, RSI, MACD, ATRが既存のシステムと正しく統合されていることを確認します。
"""

import numpy as np
import pandas as pd
import pytest
from typing import Tuple

# 移行済みの指標クラス
from app.services.indicators.technical_indicators.trend import TrendIndicators
from app.services.indicators.technical_indicators.momentum import MomentumIndicators
from app.services.indicators.technical_indicators.volatility import VolatilityIndicators


class TestBasicIndicatorsIntegration:
    """基本指標統合テストクラス"""

    @pytest.fixture
    def sample_market_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """テスト用の市場データを生成"""
        np.random.seed(42)
        
        # 現実的な価格データを生成
        base_price = 100.0
        returns = np.random.normal(0, 0.02, 300)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # OHLC生成
        close = np.array(prices, dtype=np.float64)
        high = close * (1 + np.abs(np.random.normal(0, 0.01, len(close))))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, len(close))))
        open_price = np.roll(close, 1)
        open_price[0] = close[0]
        
        return open_price, high, low, close

    def test_all_basic_indicators_calculation(self, sample_market_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        """全ての基本指標が正常に計算されることを確認"""
        open_price, high, low, close = sample_market_data
        
        # SMA計算
        sma_20 = TrendIndicators.sma(close, 20)
        assert isinstance(sma_20, np.ndarray)
        assert len(sma_20) == len(close)
        assert not np.isnan(sma_20[-10:]).all()
        
        # EMA計算
        ema_20 = TrendIndicators.ema(close, 20)
        assert isinstance(ema_20, np.ndarray)
        assert len(ema_20) == len(close)
        assert not np.isnan(ema_20[-10:]).all()
        
        # RSI計算
        rsi_14 = MomentumIndicators.rsi(close, 14)
        assert isinstance(rsi_14, np.ndarray)
        assert len(rsi_14) == len(close)
        assert not np.isnan(rsi_14[-10:]).all()
        
        # MACD計算
        macd, signal, histogram = MomentumIndicators.macd(close, 12, 26, 9)
        assert isinstance(macd, np.ndarray)
        assert isinstance(signal, np.ndarray)
        assert isinstance(histogram, np.ndarray)
        assert len(macd) == len(close)
        assert len(signal) == len(close)
        assert len(histogram) == len(close)
        
        # ATR計算
        atr_14 = VolatilityIndicators.atr(high, low, close, 14)
        assert isinstance(atr_14, np.ndarray)
        assert len(atr_14) == len(close)
        assert not np.isnan(atr_14[-10:]).all()

    def test_indicators_value_ranges(self, sample_market_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        """指標の値域が適切であることを確認"""
        open_price, high, low, close = sample_market_data
        
        # RSIは0-100の範囲
        rsi_14 = MomentumIndicators.rsi(close, 14)
        valid_rsi = rsi_14[~np.isnan(rsi_14)]
        assert np.all(valid_rsi >= 0), "RSI値が0未満です"
        assert np.all(valid_rsi <= 100), "RSI値が100を超えています"
        
        # ATRは正の値
        atr_14 = VolatilityIndicators.atr(high, low, close, 14)
        valid_atr = atr_14[~np.isnan(atr_14)]
        assert np.all(valid_atr >= 0), "ATR値に負の値があります"

    def test_indicators_consistency(self, sample_market_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        """指標間の一貫性を確認"""
        open_price, high, low, close = sample_market_data
        
        # SMAとEMAの関係（通常は異なる値になる）
        sma_20 = TrendIndicators.sma(close, 20)
        ema_20 = TrendIndicators.ema(close, 20)
        
        valid_mask = ~(np.isnan(sma_20) | np.isnan(ema_20))
        if np.any(valid_mask):
            # 最後の10個の値で違いがあることを確認
            last_10_mask = valid_mask[-10:]
            if np.any(last_10_mask):
                sma_last = sma_20[-10:][last_10_mask]
                ema_last = ema_20[-10:][last_10_mask]
                assert not np.allclose(sma_last, ema_last, rtol=1e-6), "SMAとEMAが同じ値になっています"

    def test_indicators_with_minimal_data(self):
        """最小限のデータでの指標計算テスト"""
        # 50個のデータポイント
        close = np.array([100 + i * 0.1 for i in range(50)], dtype=float)
        high = close + 1
        low = close - 1
        
        # 各指標が計算できることを確認
        sma_10 = TrendIndicators.sma(close, 10)
        ema_10 = TrendIndicators.ema(close, 10)
        rsi_14 = MomentumIndicators.rsi(close, 14)
        macd, signal, histogram = MomentumIndicators.macd(close, 12, 26, 9)
        atr_14 = VolatilityIndicators.atr(high, low, close, 14)
        
        # 全て適切な長さであることを確認
        assert len(sma_10) == len(close)
        assert len(ema_10) == len(close)
        assert len(rsi_14) == len(close)
        assert len(macd) == len(close)
        assert len(atr_14) == len(close)

    def test_indicators_performance_baseline(self, sample_market_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        """指標計算のパフォーマンスベースライン"""
        import time
        
        open_price, high, low, close = sample_market_data
        iterations = 10
        
        start_time = time.time()
        for _ in range(iterations):
            # 全ての基本指標を計算
            TrendIndicators.sma(close, 20)
            TrendIndicators.ema(close, 20)
            MomentumIndicators.rsi(close, 14)
            MomentumIndicators.macd(close, 12, 26, 9)
            VolatilityIndicators.atr(high, low, close, 14)
        
        total_time = time.time() - start_time
        avg_time = total_time / iterations
        
        print(f"\n基本指標計算パフォーマンス:")
        print(f"平均実行時間: {avg_time:.4f}秒")
        print(f"データポイント数: {len(close)}")
        print(f"1データポイントあたり: {avg_time/len(close)*1000:.4f}ms")
        
        # パフォーマンスが合理的な範囲内であることを確認
        assert avg_time < 1.0, f"基本指標計算が遅すぎます: {avg_time:.4f}秒"

    def test_indicators_memory_usage(self, sample_market_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        """指標計算のメモリ使用量テスト"""
        import sys
        
        open_price, high, low, close = sample_market_data
        
        # 計算前のメモリ使用量
        initial_size = sys.getsizeof(close)
        
        # 指標計算
        sma_20 = TrendIndicators.sma(close, 20)
        ema_20 = TrendIndicators.ema(close, 20)
        rsi_14 = MomentumIndicators.rsi(close, 14)
        macd, signal, histogram = MomentumIndicators.macd(close, 12, 26, 9)
        atr_14 = VolatilityIndicators.atr(high, low, close, 14)
        
        # 結果のメモリ使用量
        total_result_size = (
            sys.getsizeof(sma_20) + sys.getsizeof(ema_20) + 
            sys.getsizeof(rsi_14) + sys.getsizeof(macd) + 
            sys.getsizeof(signal) + sys.getsizeof(histogram) + 
            sys.getsizeof(atr_14)
        )
        
        print(f"\nメモリ使用量:")
        print(f"入力データ: {initial_size} bytes")
        print(f"結果データ: {total_result_size} bytes")
        print(f"比率: {total_result_size/initial_size:.2f}x")

    def test_indicators_error_propagation(self):
        """エラー伝播のテスト"""
        from app.services.indicators.pandas_ta_utils import PandasTAError
        
        # 不正なデータでエラーが適切に発生することを確認
        empty_data = np.array([])
        
        with pytest.raises(PandasTAError):
            TrendIndicators.sma(empty_data, 10)
        
        with pytest.raises(PandasTAError):
            TrendIndicators.ema(empty_data, 10)
        
        with pytest.raises(PandasTAError):
            MomentumIndicators.rsi(empty_data, 14)
        
        with pytest.raises(PandasTAError):
            MomentumIndicators.macd(empty_data, 12, 26, 9)
        
        with pytest.raises(PandasTAError):
            VolatilityIndicators.atr(empty_data, empty_data, empty_data, 14)

    def test_indicators_with_real_market_patterns(self):
        """実際の市場パターンでのテスト"""
        # 上昇トレンド
        uptrend_data = np.array([100 + i * 0.5 for i in range(100)], dtype=float)
        
        sma_20 = TrendIndicators.sma(uptrend_data, 20)
        ema_20 = TrendIndicators.ema(uptrend_data, 20)
        rsi_14 = MomentumIndicators.rsi(uptrend_data, 14)
        
        # 上昇トレンドでは移動平均も上昇傾向
        valid_sma = sma_20[~np.isnan(sma_20)]
        if len(valid_sma) > 10:
            assert valid_sma[-1] > valid_sma[-10], "上昇トレンドでSMAが上昇していません"
        
        # 下降トレンド
        downtrend_data = np.array([100 - i * 0.5 for i in range(100)], dtype=float)
        
        sma_20 = TrendIndicators.sma(downtrend_data, 20)
        valid_sma = sma_20[~np.isnan(sma_20)]
        if len(valid_sma) > 10:
            assert valid_sma[-1] < valid_sma[-10], "下降トレンドでSMAが下降していません"

    def test_indicators_nan_handling(self, sample_market_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        """NaN値の適切な処理テスト"""
        open_price, high, low, close = sample_market_data
        
        # 一部にNaN値を挿入
        close_with_nan = close.copy()
        close_with_nan[100:110] = np.nan
        
        # 指標計算がエラーなく実行されることを確認
        sma_20 = TrendIndicators.sma(close_with_nan, 20)
        ema_20 = TrendIndicators.ema(close_with_nan, 20)
        rsi_14 = MomentumIndicators.rsi(close_with_nan, 14)
        
        # 結果が適切な長さであることを確認
        assert len(sma_20) == len(close_with_nan)
        assert len(ema_20) == len(close_with_nan)
        assert len(rsi_14) == len(close_with_nan)


if __name__ == "__main__":
    # 統合テストを実行
    pytest.main([__file__, "-v", "-s"])
