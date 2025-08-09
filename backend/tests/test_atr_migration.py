"""
ATR指標のTA-libからpandas-taへの移行テスト

このテストでは以下を確認します：
1. TA-libとpandas-taの計算結果の一致性
2. 複数入力（高値、安値、終値）の処理
3. backtesting.py互換性
4. エラーハンドリング
"""

import numpy as np
import pandas as pd
import pytest
import talib
from typing import Tuple

# 既存のTA-lib実装
from app.services.indicators.technical_indicators.volatility import VolatilityIndicators

# 新しいpandas-ta実装
from app.services.indicators.pandas_ta_utils import pandas_ta_atr


class TestATRMigration:
    """ATR移行テストクラス"""

    @pytest.fixture
    def sample_ohlc_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """テスト用のOHLC価格データを生成"""
        np.random.seed(42)
        
        # 現実的な価格データを生成
        base_price = 100.0
        returns = np.random.normal(0, 0.02, 200)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # OHLC生成
        close = np.array(prices, dtype=np.float64)
        high = close * (1 + np.abs(np.random.normal(0, 0.01, len(close))))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, len(close))))
        open_price = np.roll(close, 1)
        open_price[0] = close[0]
        
        # pandas DataFrameも作成
        dates = pd.date_range('2023-01-01', periods=len(close), freq='D')
        ohlc_df = pd.DataFrame({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close
        }, index=dates)
        
        return high, low, close, ohlc_df

    def test_atr_calculation_consistency(self, sample_ohlc_data: Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]):
        """TA-libとpandas-taのATR計算結果の一致性テスト"""
        high, low, close, ohlc_df = sample_ohlc_data
        period = 14
        
        # TA-lib版ATR
        talib_atr = VolatilityIndicators.atr(high, low, close, period)
        
        # pandas-ta版ATR
        pandas_ta_atr_result = pandas_ta_atr(high, low, close, period)
        
        # 結果の比較（NaN以外の部分）
        valid_mask = ~(np.isnan(talib_atr) | np.isnan(pandas_ta_atr_result))
        
        np.testing.assert_allclose(
            talib_atr[valid_mask],
            pandas_ta_atr_result[valid_mask],
            rtol=1e-10,
            err_msg="TA-libとpandas-taのATR結果が一致しません"
        )

    def test_atr_with_pandas_series_input(self, sample_ohlc_data: Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]):
        """pandas Series入力でのATRテスト"""
        high, low, close, ohlc_df = sample_ohlc_data
        period = 14
        
        # TA-lib版（numpy配列）
        talib_atr = VolatilityIndicators.atr(high, low, close, period)
        
        # pandas-ta版（pandas Series）
        pandas_ta_atr_result = pandas_ta_atr(
            ohlc_df['High'], ohlc_df['Low'], ohlc_df['Close'], period
        )
        
        # 結果の比較
        valid_mask = ~(np.isnan(talib_atr) | np.isnan(pandas_ta_atr_result))
        
        np.testing.assert_allclose(
            talib_atr[valid_mask],
            pandas_ta_atr_result[valid_mask],
            rtol=1e-10,
            err_msg="pandas Series入力でのATR結果が一致しません"
        )

    def test_atr_different_periods(self, sample_ohlc_data: Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]):
        """異なる期間でのATRテスト"""
        high, low, close, _ = sample_ohlc_data
        periods = [7, 14, 21, 30]
        
        for period in periods:
            # TA-lib版
            talib_atr = VolatilityIndicators.atr(high, low, close, period)
            
            # pandas-ta版
            pandas_ta_atr_result = pandas_ta_atr(high, low, close, period)
            
            # 結果の比較
            valid_mask = ~(np.isnan(talib_atr) | np.isnan(pandas_ta_atr_result))
            
            np.testing.assert_allclose(
                talib_atr[valid_mask],
                pandas_ta_atr_result[valid_mask],
                rtol=1e-10,
                err_msg=f"期間{period}でのATR結果が一致しません"
            )

    def test_atr_positive_values(self, sample_ohlc_data: Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]):
        """ATR値が正の値であることの確認テスト"""
        high, low, close, _ = sample_ohlc_data
        period = 14
        
        # pandas-ta版ATR
        result = pandas_ta_atr(high, low, close, period)
        
        # NaN以外の値を取得
        valid_values = result[~np.isnan(result)]
        
        # ATRは常に正の値であることを確認
        assert np.all(valid_values >= 0), "ATR値に負の値があります"

    def test_atr_backtesting_compatibility(self, sample_ohlc_data: Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]):
        """backtesting.py互換性テスト"""
        high, low, close, _ = sample_ohlc_data
        period = 14
        
        # pandas-ta版ATR
        result = pandas_ta_atr(high, low, close, period)
        
        # numpy配列として返されることを確認
        assert isinstance(result, np.ndarray), "結果がnumpy配列ではありません"
        
        # 長さが元データと一致することを確認
        assert len(result) == len(high), "結果の長さが元データと一致しません"
        
        # 有効な値が存在することを確認
        assert not np.isnan(result[-10:]).all(), "最後の10個の値が全てNaNです"

    def test_atr_error_handling(self):
        """エラーハンドリングテスト"""
        from app.services.indicators.pandas_ta_utils import PandasTAError
        
        # 空のデータ
        with pytest.raises(PandasTAError):
            pandas_ta_atr(np.array([]), np.array([]), np.array([]), 14)
        
        # 期間がデータ長より長い
        short_high = np.array([1.0, 2.0, 3.0])
        short_low = np.array([0.5, 1.5, 2.5])
        short_close = np.array([0.8, 1.8, 2.8])
        with pytest.raises(PandasTAError):
            pandas_ta_atr(short_high, short_low, short_close, 20)
        
        # 長さが一致しないデータ
        with pytest.raises(PandasTAError):
            pandas_ta_atr(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.5, 1.5]),  # 長さが異なる
                np.array([0.8, 1.8, 2.8]),
                5
            )

    def test_atr_with_nan_values(self, sample_ohlc_data: Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]):
        """NaN値を含むデータでのATRテスト"""
        high, low, close, _ = sample_ohlc_data
        period = 14
        
        # 一部にNaN値を挿入
        high_with_nan = high.copy()
        low_with_nan = low.copy()
        close_with_nan = close.copy()
        
        high_with_nan[50:55] = np.nan
        low_with_nan[50:55] = np.nan
        close_with_nan[50:55] = np.nan
        
        # pandas-ta版ATR（NaN値の処理を確認）
        result = pandas_ta_atr(high_with_nan, low_with_nan, close_with_nan, period)
        
        # 結果がnumpy配列であることを確認
        assert isinstance(result, np.ndarray)
        
        # NaN以外の部分で有効な値が存在することを確認
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0, "有効な値が存在しません"

    def test_atr_edge_cases(self):
        """エッジケースのテスト"""
        # 価格変動がない場合（ATR = 0に近い）
        stable_price = 100.0
        stable_high = np.array([stable_price] * 20)
        stable_low = np.array([stable_price] * 20)
        stable_close = np.array([stable_price] * 20)
        
        result = pandas_ta_atr(stable_high, stable_low, stable_close, 14)
        
        # 最後の値は0に近いことを確認
        last_valid = result[~np.isnan(result)][-1]
        assert last_valid < 0.01, f"価格変動がない場合のATRが期待より大きいです: {last_valid}"

    def test_atr_price_consistency(self):
        """価格の整合性テスト"""
        # 高値 >= 終値 >= 安値の関係を満たすデータ
        close = np.array([100, 102, 101, 103, 105], dtype=float)
        high = close + 1  # 高値は終値より高い
        low = close - 1   # 安値は終値より低い
        
        result = pandas_ta_atr(high, low, close, 3)
        
        # 結果が計算されることを確認
        assert isinstance(result, np.ndarray)
        assert len(result) == len(close)

    def test_atr_performance_comparison(self, sample_ohlc_data: Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]):
        """パフォーマンス比較テスト（参考用）"""
        import time
        
        high, low, close, _ = sample_ohlc_data
        period = 14
        iterations = 100
        
        # TA-lib版のパフォーマンス測定
        start_time = time.time()
        for _ in range(iterations):
            VolatilityIndicators.atr(high, low, close, period)
        talib_time = time.time() - start_time
        
        # pandas-ta版のパフォーマンス測定
        start_time = time.time()
        for _ in range(iterations):
            pandas_ta_atr(high, low, close, period)
        pandas_ta_time = time.time() - start_time
        
        print(f"\nパフォーマンス比較 (ATR, {iterations}回実行):")
        print(f"TA-lib: {talib_time:.4f}秒")
        print(f"pandas-ta: {pandas_ta_time:.4f}秒")
        print(f"比率: {pandas_ta_time/talib_time:.2f}x")

    def test_direct_talib_vs_pandas_ta_comparison(self, sample_ohlc_data: Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]):
        """直接的なTA-libとpandas-ta比較"""
        high, low, close, _ = sample_ohlc_data
        period = 14
        
        # 直接TA-lib呼び出し
        direct_talib_atr = talib.ATR(high, low, close, timeperiod=period)
        
        # pandas-ta版
        pandas_ta_atr_result = pandas_ta_atr(high, low, close, period)
        
        # 結果の比較
        valid_mask = ~(np.isnan(direct_talib_atr) | np.isnan(pandas_ta_atr_result))
        
        np.testing.assert_allclose(
            direct_talib_atr[valid_mask],
            pandas_ta_atr_result[valid_mask],
            rtol=1e-10,
            err_msg="直接TA-lib呼び出しとpandas-taの結果が一致しません"
        )


if __name__ == "__main__":
    # 基本的なテストを実行
    pytest.main([__file__, "-v"])
