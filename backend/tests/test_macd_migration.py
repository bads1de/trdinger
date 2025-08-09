"""
MACD指標のTA-libからpandas-taへの移行テスト

このテストでは以下を確認します：
1. TA-libとpandas-taの計算結果の一致性
2. 複数戻り値（MACD線、シグナル線、ヒストグラム）の処理
3. backtesting.py互換性
4. エラーハンドリング
"""

import numpy as np
import pandas as pd
import pytest
import talib
from typing import Tuple

# 既存のTA-lib実装
from app.services.indicators.technical_indicators.momentum import MomentumIndicators

# 新しいpandas-ta実装
from app.services.indicators.pandas_ta_utils import pandas_ta_macd


class TestMACDMigration:
    """MACD移行テストクラス"""

    @pytest.fixture
    def sample_price_data(self) -> Tuple[np.ndarray, pd.Series]:
        """テスト用の価格データを生成"""
        np.random.seed(42)
        
        # より長い価格データを生成（MACDの計算に適している）
        base_price = 100.0
        returns = np.random.normal(0, 0.02, 300)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        price_array = np.array(prices, dtype=np.float64)
        
        # pandas Seriesも作成
        dates = pd.date_range('2023-01-01', periods=len(prices), freq='D')
        price_series = pd.Series(price_array, index=dates)
        
        return price_array, price_series

    def test_macd_calculation_consistency(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """TA-libとpandas-taのMACD計算結果の一致性テスト"""
        price_array, price_series = sample_price_data
        fast, slow, signal = 12, 26, 9
        
        # TA-lib版MACD
        talib_macd, talib_signal, talib_hist = MomentumIndicators.macd(
            price_array, fast, slow, signal
        )
        
        # pandas-ta版MACD
        pandas_ta_macd_result, pandas_ta_signal_result, pandas_ta_hist_result = pandas_ta_macd(
            price_array, fast, slow, signal
        )
        
        # MACD線の比較
        valid_mask = ~(np.isnan(talib_macd) | np.isnan(pandas_ta_macd_result))
        np.testing.assert_allclose(
            talib_macd[valid_mask],
            pandas_ta_macd_result[valid_mask],
            rtol=1e-10,
            err_msg="MACD線の結果が一致しません"
        )
        
        # シグナル線の比較
        valid_mask = ~(np.isnan(talib_signal) | np.isnan(pandas_ta_signal_result))
        np.testing.assert_allclose(
            talib_signal[valid_mask],
            pandas_ta_signal_result[valid_mask],
            rtol=1e-10,
            err_msg="MACDシグナル線の結果が一致しません"
        )
        
        # ヒストグラムの比較
        valid_mask = ~(np.isnan(talib_hist) | np.isnan(pandas_ta_hist_result))
        np.testing.assert_allclose(
            talib_hist[valid_mask],
            pandas_ta_hist_result[valid_mask],
            rtol=1e-10,
            err_msg="MACDヒストグラムの結果が一致しません"
        )

    def test_macd_with_pandas_series_input(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """pandas Series入力でのMACDテスト"""
        price_array, price_series = sample_price_data
        fast, slow, signal = 12, 26, 9
        
        # TA-lib版（numpy配列）
        talib_macd, talib_signal, talib_hist = MomentumIndicators.macd(
            price_array, fast, slow, signal
        )
        
        # pandas-ta版（pandas Series）
        pandas_ta_macd_result, pandas_ta_signal_result, pandas_ta_hist_result = pandas_ta_macd(
            price_series, fast, slow, signal
        )
        
        # 結果の比較
        valid_mask = ~(np.isnan(talib_macd) | np.isnan(pandas_ta_macd_result))
        np.testing.assert_allclose(
            talib_macd[valid_mask],
            pandas_ta_macd_result[valid_mask],
            rtol=1e-10,
            err_msg="pandas Series入力でのMACD結果が一致しません"
        )

    def test_macd_different_parameters(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """異なるパラメータでのMACDテスト"""
        price_array, _ = sample_price_data
        parameter_sets = [
            (5, 10, 3),
            (12, 26, 9),
            (8, 21, 5),
            (15, 30, 10)
        ]
        
        for fast, slow, signal in parameter_sets:
            # TA-lib版
            talib_macd, talib_signal, talib_hist = MomentumIndicators.macd(
                price_array, fast, slow, signal
            )
            
            # pandas-ta版
            pandas_ta_macd_result, pandas_ta_signal_result, pandas_ta_hist_result = pandas_ta_macd(
                price_array, fast, slow, signal
            )
            
            # MACD線の比較
            valid_mask = ~(np.isnan(talib_macd) | np.isnan(pandas_ta_macd_result))
            np.testing.assert_allclose(
                talib_macd[valid_mask],
                pandas_ta_macd_result[valid_mask],
                rtol=1e-10,
                err_msg=f"パラメータ({fast},{slow},{signal})でのMACD結果が一致しません"
            )

    def test_macd_return_types(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """MACD戻り値の型テスト"""
        price_array, _ = sample_price_data
        fast, slow, signal = 12, 26, 9
        
        # pandas-ta版MACD
        result = pandas_ta_macd(price_array, fast, slow, signal)
        
        # タプルとして返されることを確認
        assert isinstance(result, tuple), "結果がタプルではありません"
        assert len(result) == 3, "戻り値の数が3つではありません"
        
        macd_line, signal_line, histogram = result
        
        # 各要素がnumpy配列であることを確認
        assert isinstance(macd_line, np.ndarray), "MACD線がnumpy配列ではありません"
        assert isinstance(signal_line, np.ndarray), "シグナル線がnumpy配列ではありません"
        assert isinstance(histogram, np.ndarray), "ヒストグラムがnumpy配列ではありません"
        
        # 長さが元データと一致することを確認
        assert len(macd_line) == len(price_array), "MACD線の長さが元データと一致しません"
        assert len(signal_line) == len(price_array), "シグナル線の長さが元データと一致しません"
        assert len(histogram) == len(price_array), "ヒストグラムの長さが元データと一致しません"

    def test_macd_backtesting_compatibility(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """backtesting.py互換性テスト"""
        price_array, _ = sample_price_data
        fast, slow, signal = 12, 26, 9
        
        # pandas-ta版MACD
        macd_line, signal_line, histogram = pandas_ta_macd(price_array, fast, slow, signal)
        
        # 有効な値が存在することを確認
        assert not np.isnan(macd_line[-10:]).all(), "MACD線の最後の10個の値が全てNaNです"
        assert not np.isnan(signal_line[-10:]).all(), "シグナル線の最後の10個の値が全てNaNです"
        assert not np.isnan(histogram[-10:]).all(), "ヒストグラムの最後の10個の値が全てNaNです"

    def test_macd_error_handling(self):
        """エラーハンドリングテスト"""
        from app.services.indicators.pandas_ta_utils import PandasTAError
        
        # 空のデータ
        with pytest.raises(PandasTAError):
            pandas_ta_macd(np.array([]), 12, 26, 9)
        
        # 期間がデータ長より長い
        short_data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(PandasTAError):
            pandas_ta_macd(short_data, 12, 26, 9)
        
        # 全てNaNのデータ
        nan_data = np.array([np.nan] * 50)
        with pytest.raises(PandasTAError):
            pandas_ta_macd(nan_data, 12, 26, 9)

    def test_macd_with_nan_values(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """NaN値を含むデータでのMACDテスト"""
        price_array, _ = sample_price_data
        fast, slow, signal = 12, 26, 9
        
        # 一部にNaN値を挿入
        price_with_nan = price_array.copy()
        price_with_nan[100:105] = np.nan
        
        # pandas-ta版MACD（NaN値の処理を確認）
        macd_line, signal_line, histogram = pandas_ta_macd(price_with_nan, fast, slow, signal)
        
        # 結果がnumpy配列であることを確認
        assert isinstance(macd_line, np.ndarray)
        assert isinstance(signal_line, np.ndarray)
        assert isinstance(histogram, np.ndarray)
        
        # NaN以外の部分で有効な値が存在することを確認
        valid_macd = macd_line[~np.isnan(macd_line)]
        assert len(valid_macd) > 0, "MACD線に有効な値が存在しません"

    def test_macd_histogram_calculation(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """MACDヒストグラムの計算確認テスト"""
        price_array, _ = sample_price_data
        fast, slow, signal = 12, 26, 9
        
        # pandas-ta版MACD
        macd_line, signal_line, histogram = pandas_ta_macd(price_array, fast, slow, signal)
        
        # ヒストグラム = MACD線 - シグナル線 であることを確認
        valid_mask = ~(np.isnan(macd_line) | np.isnan(signal_line) | np.isnan(histogram))
        
        if np.any(valid_mask):
            calculated_histogram = macd_line[valid_mask] - signal_line[valid_mask]
            np.testing.assert_allclose(
                histogram[valid_mask],
                calculated_histogram,
                rtol=1e-10,
                err_msg="ヒストグラムの計算が正しくありません"
            )

    def test_macd_performance_comparison(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """パフォーマンス比較テスト（参考用）"""
        import time
        
        price_array, _ = sample_price_data
        fast, slow, signal = 12, 26, 9
        iterations = 50
        
        # TA-lib版のパフォーマンス測定
        start_time = time.time()
        for _ in range(iterations):
            MomentumIndicators.macd(price_array, fast, slow, signal)
        talib_time = time.time() - start_time
        
        # pandas-ta版のパフォーマンス測定
        start_time = time.time()
        for _ in range(iterations):
            pandas_ta_macd(price_array, fast, slow, signal)
        pandas_ta_time = time.time() - start_time
        
        print(f"\nパフォーマンス比較 (MACD, {iterations}回実行):")
        print(f"TA-lib: {talib_time:.4f}秒")
        print(f"pandas-ta: {pandas_ta_time:.4f}秒")
        print(f"比率: {pandas_ta_time/talib_time:.2f}x")

    def test_direct_talib_vs_pandas_ta_comparison(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """直接的なTA-libとpandas-ta比較"""
        price_array, _ = sample_price_data
        fast, slow, signal = 12, 26, 9
        
        # 直接TA-lib呼び出し
        direct_talib_macd, direct_talib_signal, direct_talib_hist = talib.MACD(
            price_array, fastperiod=fast, slowperiod=slow, signalperiod=signal
        )
        
        # pandas-ta版
        pandas_ta_macd_result, pandas_ta_signal_result, pandas_ta_hist_result = pandas_ta_macd(
            price_array, fast, slow, signal
        )
        
        # 結果の比較
        valid_mask = ~(np.isnan(direct_talib_macd) | np.isnan(pandas_ta_macd_result))
        np.testing.assert_allclose(
            direct_talib_macd[valid_mask],
            pandas_ta_macd_result[valid_mask],
            rtol=1e-10,
            err_msg="直接TA-lib呼び出しとpandas-taのMACD結果が一致しません"
        )


if __name__ == "__main__":
    # 基本的なテストを実行
    pytest.main([__file__, "-v"])
