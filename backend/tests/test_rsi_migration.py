"""
RSI指標のTA-libからpandas-taへの移行テスト

このテストでは以下を確認します：
1. TA-libとpandas-taの計算結果の一致性
2. backtesting.py互換性
3. エラーハンドリング
4. RSI特有の値域（0-100）の確認
"""

import numpy as np
import pandas as pd
import pytest
import talib
from typing import Tuple

# 既存のTA-lib実装
from app.services.indicators.technical_indicators.momentum import MomentumIndicators

# 新しいpandas-ta実装
from app.services.indicators.pandas_ta_utils import pandas_ta_rsi


class TestRSIMigration:
    """RSI移行テストクラス"""

    @pytest.fixture
    def sample_price_data(self) -> Tuple[np.ndarray, pd.Series]:
        """テスト用の価格データを生成"""
        np.random.seed(42)
        
        # より変動の大きい価格データを生成（RSIの計算に適している）
        base_price = 100.0
        returns = np.random.normal(0, 0.03, 200)  # より大きな変動
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        price_array = np.array(prices, dtype=np.float64)
        
        # pandas Seriesも作成
        dates = pd.date_range('2023-01-01', periods=len(prices), freq='D')
        price_series = pd.Series(price_array, index=dates)
        
        return price_array, price_series

    def test_rsi_calculation_consistency(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """TA-libとpandas-taのRSI計算結果の一致性テスト"""
        price_array, price_series = sample_price_data
        period = 14
        
        # TA-lib版RSI
        talib_rsi = MomentumIndicators.rsi(price_array, period)
        
        # pandas-ta版RSI
        pandas_ta_rsi_result = pandas_ta_rsi(price_array, period)
        
        # 結果の比較（NaN以外の部分）
        valid_mask = ~(np.isnan(talib_rsi) | np.isnan(pandas_ta_rsi_result))
        
        np.testing.assert_allclose(
            talib_rsi[valid_mask],
            pandas_ta_rsi_result[valid_mask],
            rtol=1e-10,
            err_msg="TA-libとpandas-taのRSI結果が一致しません"
        )

    def test_rsi_with_pandas_series_input(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """pandas Series入力でのRSIテスト"""
        price_array, price_series = sample_price_data
        period = 14
        
        # TA-lib版（numpy配列）
        talib_rsi = MomentumIndicators.rsi(price_array, period)
        
        # pandas-ta版（pandas Series）
        pandas_ta_rsi_result = pandas_ta_rsi(price_series, period)
        
        # 結果の比較
        valid_mask = ~(np.isnan(talib_rsi) | np.isnan(pandas_ta_rsi_result))
        
        np.testing.assert_allclose(
            talib_rsi[valid_mask],
            pandas_ta_rsi_result[valid_mask],
            rtol=1e-10,
            err_msg="pandas Series入力でのRSI結果が一致しません"
        )

    def test_rsi_different_periods(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """異なる期間でのRSIテスト"""
        price_array, _ = sample_price_data
        periods = [7, 14, 21, 30]
        
        for period in periods:
            # TA-lib版
            talib_rsi = MomentumIndicators.rsi(price_array, period)
            
            # pandas-ta版
            pandas_ta_rsi_result = pandas_ta_rsi(price_array, period)
            
            # 結果の比較
            valid_mask = ~(np.isnan(talib_rsi) | np.isnan(pandas_ta_rsi_result))
            
            np.testing.assert_allclose(
                talib_rsi[valid_mask],
                pandas_ta_rsi_result[valid_mask],
                rtol=1e-10,
                err_msg=f"期間{period}でのRSI結果が一致しません"
            )

    def test_rsi_value_range(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """RSI値域（0-100）の確認テスト"""
        price_array, _ = sample_price_data
        period = 14
        
        # pandas-ta版RSI
        result = pandas_ta_rsi(price_array, period)
        
        # NaN以外の値を取得
        valid_values = result[~np.isnan(result)]
        
        # RSIは0-100の範囲内であることを確認
        assert np.all(valid_values >= 0), "RSI値が0未満です"
        assert np.all(valid_values <= 100), "RSI値が100を超えています"

    def test_rsi_backtesting_compatibility(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """backtesting.py互換性テスト"""
        price_array, _ = sample_price_data
        period = 14
        
        # pandas-ta版RSI
        result = pandas_ta_rsi(price_array, period)
        
        # numpy配列として返されることを確認
        assert isinstance(result, np.ndarray), "結果がnumpy配列ではありません"
        
        # 長さが元データと一致することを確認
        assert len(result) == len(price_array), "結果の長さが元データと一致しません"
        
        # 有効な値が存在することを確認
        assert not np.isnan(result[-10:]).all(), "最後の10個の値が全てNaNです"

    def test_rsi_error_handling(self):
        """エラーハンドリングテスト"""
        from app.services.indicators.pandas_ta_utils import PandasTAError
        
        # 空のデータ
        with pytest.raises(PandasTAError):
            pandas_ta_rsi(np.array([]), 14)
        
        # 期間がデータ長より長い
        short_data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(PandasTAError):
            pandas_ta_rsi(short_data, 20)
        
        # 全てNaNのデータ
        nan_data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        with pytest.raises(PandasTAError):
            pandas_ta_rsi(nan_data, 3)

    def test_rsi_with_nan_values(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """NaN値を含むデータでのRSIテスト"""
        price_array, _ = sample_price_data
        period = 14
        
        # 一部にNaN値を挿入
        price_with_nan = price_array.copy()
        price_with_nan[50:55] = np.nan
        
        # pandas-ta版RSI（NaN値の処理を確認）
        result = pandas_ta_rsi(price_with_nan, period)
        
        # 結果がnumpy配列であることを確認
        assert isinstance(result, np.ndarray)
        
        # NaN以外の部分で有効な値が存在することを確認
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0, "有効な値が存在しません"

    def test_rsi_edge_cases(self):
        """エッジケースのテスト"""
        # 単調増加データ（RSI = 100に近づく）
        increasing_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 
                                   11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
        result = pandas_ta_rsi(increasing_data, 14)
        
        # 最後の値は高いRSI値になることを確認
        last_valid = result[~np.isnan(result)][-1]
        assert last_valid > 70, f"単調増加データのRSIが期待より低いです: {last_valid}"
        
        # 単調減少データ（RSI = 0に近づく）
        decreasing_data = np.array([20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0,
                                   10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        result = pandas_ta_rsi(decreasing_data, 14)
        
        # 最後の値は低いRSI値になることを確認
        last_valid = result[~np.isnan(result)][-1]
        assert last_valid < 30, f"単調減少データのRSIが期待より高いです: {last_valid}"

    def test_rsi_performance_comparison(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """パフォーマンス比較テスト（参考用）"""
        import time
        
        price_array, _ = sample_price_data
        period = 14
        iterations = 100
        
        # TA-lib版のパフォーマンス測定
        start_time = time.time()
        for _ in range(iterations):
            MomentumIndicators.rsi(price_array, period)
        talib_time = time.time() - start_time
        
        # pandas-ta版のパフォーマンス測定
        start_time = time.time()
        for _ in range(iterations):
            pandas_ta_rsi(price_array, period)
        pandas_ta_time = time.time() - start_time
        
        print(f"\nパフォーマンス比較 (RSI, {iterations}回実行):")
        print(f"TA-lib: {talib_time:.4f}秒")
        print(f"pandas-ta: {pandas_ta_time:.4f}秒")
        print(f"比率: {pandas_ta_time/talib_time:.2f}x")

    def test_direct_talib_vs_pandas_ta_comparison(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """直接的なTA-libとpandas-ta比較"""
        price_array, _ = sample_price_data
        period = 14
        
        # 直接TA-lib呼び出し
        direct_talib_rsi = talib.RSI(price_array, timeperiod=period)
        
        # pandas-ta版
        pandas_ta_rsi_result = pandas_ta_rsi(price_array, period)
        
        # 結果の比較
        valid_mask = ~(np.isnan(direct_talib_rsi) | np.isnan(pandas_ta_rsi_result))
        
        np.testing.assert_allclose(
            direct_talib_rsi[valid_mask],
            pandas_ta_rsi_result[valid_mask],
            rtol=1e-10,
            err_msg="直接TA-lib呼び出しとpandas-taの結果が一致しません"
        )

    def test_rsi_overbought_oversold_levels(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """RSIの買われすぎ・売られすぎレベルのテスト"""
        price_array, _ = sample_price_data
        period = 14
        
        # pandas-ta版RSI
        result = pandas_ta_rsi(price_array, period)
        
        # NaN以外の値を取得
        valid_values = result[~np.isnan(result)]
        
        # 買われすぎ（70以上）と売られすぎ（30以下）の値が存在することを確認
        # （データによっては存在しない場合もあるので、存在する場合のみチェック）
        overbought = valid_values[valid_values >= 70]
        oversold = valid_values[valid_values <= 30]
        
        if len(overbought) > 0:
            assert np.all(overbought <= 100), "買われすぎレベルのRSI値が100を超えています"
        
        if len(oversold) > 0:
            assert np.all(oversold >= 0), "売られすぎレベルのRSI値が0未満です"


if __name__ == "__main__":
    # 基本的なテストを実行
    pytest.main([__file__, "-v"])
