"""
EMA指標のTA-libからpandas-taへの移行テスト

このテストでは以下を確認します：
1. TA-libとpandas-taの計算結果の一致性
2. backtesting.py互換性
3. エラーハンドリング
4. パフォーマンス比較
"""

import numpy as np
import pandas as pd
import pytest
import talib
from typing import Tuple

# 既存のTA-lib実装
from app.services.indicators.technical_indicators.trend import TrendIndicators

# 新しいpandas-ta実装
from app.services.indicators.pandas_ta_utils import pandas_ta_ema


class TestEMAMigration:
    """EMA移行テストクラス"""

    @pytest.fixture
    def sample_price_data(self) -> Tuple[np.ndarray, pd.Series]:
        """テスト用の価格データを生成"""
        np.random.seed(42)
        
        # 現実的な価格データを生成
        base_price = 100.0
        returns = np.random.normal(0, 0.02, 200)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        price_array = np.array(prices, dtype=np.float64)
        
        # pandas Seriesも作成
        dates = pd.date_range('2023-01-01', periods=len(prices), freq='D')
        price_series = pd.Series(price_array, index=dates)
        
        return price_array, price_series

    def test_ema_calculation_consistency(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """TA-libとpandas-taのEMA計算結果の一致性テスト"""
        price_array, price_series = sample_price_data
        period = 20
        
        # TA-lib版EMA
        talib_ema = TrendIndicators.ema(price_array, period)
        
        # pandas-ta版EMA
        pandas_ta_ema_result = pandas_ta_ema(price_array, period)
        
        # 結果の比較（NaN以外の部分）
        valid_mask = ~(np.isnan(talib_ema) | np.isnan(pandas_ta_ema_result))
        
        np.testing.assert_allclose(
            talib_ema[valid_mask],
            pandas_ta_ema_result[valid_mask],
            rtol=1e-10,
            err_msg="TA-libとpandas-taのEMA結果が一致しません"
        )

    def test_ema_with_pandas_series_input(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """pandas Series入力でのEMAテスト"""
        price_array, price_series = sample_price_data
        period = 20
        
        # TA-lib版（numpy配列）
        talib_ema = TrendIndicators.ema(price_array, period)
        
        # pandas-ta版（pandas Series）
        pandas_ta_ema_result = pandas_ta_ema(price_series, period)
        
        # 結果の比較
        valid_mask = ~(np.isnan(talib_ema) | np.isnan(pandas_ta_ema_result))
        
        np.testing.assert_allclose(
            talib_ema[valid_mask],
            pandas_ta_ema_result[valid_mask],
            rtol=1e-10,
            err_msg="pandas Series入力でのEMA結果が一致しません"
        )

    def test_ema_different_periods(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """異なる期間でのEMAテスト"""
        price_array, _ = sample_price_data
        periods = [5, 10, 20, 50]
        
        for period in periods:
            # TA-lib版
            talib_ema = TrendIndicators.ema(price_array, period)
            
            # pandas-ta版
            pandas_ta_ema_result = pandas_ta_ema(price_array, period)
            
            # 結果の比較
            valid_mask = ~(np.isnan(talib_ema) | np.isnan(pandas_ta_ema_result))
            
            np.testing.assert_allclose(
                talib_ema[valid_mask],
                pandas_ta_ema_result[valid_mask],
                rtol=1e-10,
                err_msg=f"期間{period}でのEMA結果が一致しません"
            )

    def test_ema_backtesting_compatibility(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """backtesting.py互換性テスト"""
        price_array, _ = sample_price_data
        period = 20
        
        # pandas-ta版EMA
        result = pandas_ta_ema(price_array, period)
        
        # numpy配列として返されることを確認
        assert isinstance(result, np.ndarray), "結果がnumpy配列ではありません"
        
        # 長さが元データと一致することを確認
        assert len(result) == len(price_array), "結果の長さが元データと一致しません"
        
        # 有効な値が存在することを確認
        assert not np.isnan(result[-10:]).all(), "最後の10個の値が全てNaNです"

    def test_ema_error_handling(self):
        """エラーハンドリングテスト"""
        from app.services.indicators.pandas_ta_utils import PandasTAError
        
        # 空のデータ
        with pytest.raises(PandasTAError):
            pandas_ta_ema(np.array([]), 10)
        
        # 期間がデータ長より長い
        short_data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(PandasTAError):
            pandas_ta_ema(short_data, 10)
        
        # 全てNaNのデータ
        nan_data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        with pytest.raises(PandasTAError):
            pandas_ta_ema(nan_data, 3)

    def test_ema_with_nan_values(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """NaN値を含むデータでのEMAテスト"""
        price_array, _ = sample_price_data
        period = 20
        
        # 一部にNaN値を挿入
        price_with_nan = price_array.copy()
        price_with_nan[50:55] = np.nan
        
        # pandas-ta版EMA（NaN値の処理を確認）
        result = pandas_ta_ema(price_with_nan, period)
        
        # 結果がnumpy配列であることを確認
        assert isinstance(result, np.ndarray)
        
        # NaN以外の部分で有効な値が存在することを確認
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0, "有効な値が存在しません"

    def test_ema_edge_cases(self):
        """エッジケースのテスト"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # 最小期間（2）
        result = pandas_ta_ema(data, 2)
        assert isinstance(result, np.ndarray), "結果がnumpy配列ではありません"
        assert len(result) == len(data), "結果の長さが元データと一致しません"
        
        # 期間がデータ長と同じ
        result = pandas_ta_ema(data, len(data))
        assert isinstance(result, np.ndarray), "結果がnumpy配列ではありません"
        assert not np.isnan(result[-1]), "最後の値がNaNです"

    def test_ema_performance_comparison(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """パフォーマンス比較テスト（参考用）"""
        import time
        
        price_array, _ = sample_price_data
        period = 20
        iterations = 100
        
        # TA-lib版のパフォーマンス測定
        start_time = time.time()
        for _ in range(iterations):
            TrendIndicators.ema(price_array, period)
        talib_time = time.time() - start_time
        
        # pandas-ta版のパフォーマンス測定
        start_time = time.time()
        for _ in range(iterations):
            pandas_ta_ema(price_array, period)
        pandas_ta_time = time.time() - start_time
        
        print(f"\nパフォーマンス比較 (EMA, {iterations}回実行):")
        print(f"TA-lib: {talib_time:.4f}秒")
        print(f"pandas-ta: {pandas_ta_time:.4f}秒")
        print(f"比率: {pandas_ta_time/talib_time:.2f}x")

    def test_direct_talib_vs_pandas_ta_comparison(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """直接的なTA-libとpandas-ta比較"""
        price_array, _ = sample_price_data
        period = 20
        
        # 直接TA-lib呼び出し
        direct_talib_ema = talib.EMA(price_array, timeperiod=period)
        
        # pandas-ta版
        pandas_ta_ema_result = pandas_ta_ema(price_array, period)
        
        # 結果の比較
        valid_mask = ~(np.isnan(direct_talib_ema) | np.isnan(pandas_ta_ema_result))
        
        np.testing.assert_allclose(
            direct_talib_ema[valid_mask],
            pandas_ta_ema_result[valid_mask],
            rtol=1e-10,
            err_msg="直接TA-lib呼び出しとpandas-taの結果が一致しません"
        )

    def test_ema_vs_sma_difference(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """EMAとSMAの違いを確認するテスト"""
        price_array, _ = sample_price_data
        period = 20
        
        # EMAとSMAを計算
        ema_result = pandas_ta_ema(price_array, period)
        from app.services.indicators.pandas_ta_utils import pandas_ta_sma
        sma_result = pandas_ta_sma(price_array, period)
        
        # EMAとSMAは異なる値になることを確認
        valid_mask = ~(np.isnan(ema_result) | np.isnan(sma_result))
        
        # 最後の10個の値で違いがあることを確認
        last_10_mask = valid_mask[-10:]
        if np.any(last_10_mask):
            ema_last = ema_result[-10:][last_10_mask]
            sma_last = sma_result[-10:][last_10_mask]
            
            # EMAとSMAは通常異なる値になる
            assert not np.allclose(ema_last, sma_last, rtol=1e-6), "EMAとSMAが同じ値になっています"


if __name__ == "__main__":
    # 基本的なテストを実行
    pytest.main([__file__, "-v"])
