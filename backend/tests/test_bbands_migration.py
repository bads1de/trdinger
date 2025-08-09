"""
Bollinger Bands指標のTA-libからpandas-taへの移行テスト

このテストでは以下を確認します：
1. TA-libとpandas-taの計算結果の一致性
2. 複数戻り値（上限、中央、下限）の処理
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
from app.services.indicators.pandas_ta_utils import pandas_ta_bbands


class TestBBandsMigration:
    """Bollinger Bands移行テストクラス"""

    @pytest.fixture
    def sample_price_data(self) -> Tuple[np.ndarray, pd.Series]:
        """テスト用の価格データを生成"""
        np.random.seed(42)
        
        # より長い価格データを生成（Bollinger Bandsの計算に適している）
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

    def test_bbands_calculation_consistency(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """TA-libとpandas-taのBollinger Bands計算結果の一致性テスト"""
        price_array, price_series = sample_price_data
        period, std_dev = 20, 2.0
        
        # TA-lib版Bollinger Bands
        talib_upper, talib_middle, talib_lower = VolatilityIndicators.bollinger_bands(
            price_array, period, std_dev
        )
        
        # pandas-ta版Bollinger Bands
        pandas_ta_upper, pandas_ta_middle, pandas_ta_lower = pandas_ta_bbands(
            price_array, period, std_dev
        )
        
        # 上限バンドの比較
        valid_mask = ~(np.isnan(talib_upper) | np.isnan(pandas_ta_upper))
        np.testing.assert_allclose(
            talib_upper[valid_mask],
            pandas_ta_upper[valid_mask],
            rtol=1e-10,
            err_msg="Bollinger Bands上限の結果が一致しません"
        )
        
        # 中央線の比較
        valid_mask = ~(np.isnan(talib_middle) | np.isnan(pandas_ta_middle))
        np.testing.assert_allclose(
            talib_middle[valid_mask],
            pandas_ta_middle[valid_mask],
            rtol=1e-10,
            err_msg="Bollinger Bands中央線の結果が一致しません"
        )
        
        # 下限バンドの比較
        valid_mask = ~(np.isnan(talib_lower) | np.isnan(pandas_ta_lower))
        np.testing.assert_allclose(
            talib_lower[valid_mask],
            pandas_ta_lower[valid_mask],
            rtol=1e-10,
            err_msg="Bollinger Bands下限の結果が一致しません"
        )

    def test_bbands_with_pandas_series_input(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """pandas Series入力でのBollinger Bandsテスト"""
        price_array, price_series = sample_price_data
        period, std_dev = 20, 2.0
        
        # TA-lib版（numpy配列）
        talib_upper, talib_middle, talib_lower = VolatilityIndicators.bollinger_bands(
            price_array, period, std_dev
        )
        
        # pandas-ta版（pandas Series）
        pandas_ta_upper, pandas_ta_middle, pandas_ta_lower = pandas_ta_bbands(
            price_series, period, std_dev
        )
        
        # 結果の比較
        valid_mask = ~(np.isnan(talib_upper) | np.isnan(pandas_ta_upper))
        np.testing.assert_allclose(
            talib_upper[valid_mask],
            pandas_ta_upper[valid_mask],
            rtol=1e-10,
            err_msg="pandas Series入力でのBollinger Bands結果が一致しません"
        )

    def test_bbands_different_parameters(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """異なるパラメータでのBollinger Bandsテスト"""
        price_array, _ = sample_price_data
        parameter_sets = [
            (10, 1.5),
            (20, 2.0),
            (30, 2.5),
            (50, 1.0)
        ]
        
        for period, std_dev in parameter_sets:
            # TA-lib版
            talib_upper, talib_middle, talib_lower = VolatilityIndicators.bollinger_bands(
                price_array, period, std_dev
            )
            
            # pandas-ta版
            pandas_ta_upper, pandas_ta_middle, pandas_ta_lower = pandas_ta_bbands(
                price_array, period, std_dev
            )
            
            # 上限バンドの比較
            valid_mask = ~(np.isnan(talib_upper) | np.isnan(pandas_ta_upper))
            np.testing.assert_allclose(
                talib_upper[valid_mask],
                pandas_ta_upper[valid_mask],
                rtol=1e-10,
                err_msg=f"パラメータ({period},{std_dev})でのBollinger Bands結果が一致しません"
            )

    def test_bbands_return_types(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """Bollinger Bands戻り値の型テスト"""
        price_array, _ = sample_price_data
        period, std_dev = 20, 2.0
        
        # pandas-ta版Bollinger Bands
        result = pandas_ta_bbands(price_array, period, std_dev)
        
        # タプルとして返されることを確認
        assert isinstance(result, tuple), "結果がタプルではありません"
        assert len(result) == 3, "戻り値の数が3つではありません"
        
        upper, middle, lower = result
        
        # 各要素がnumpy配列であることを確認
        assert isinstance(upper, np.ndarray), "上限バンドがnumpy配列ではありません"
        assert isinstance(middle, np.ndarray), "中央線がnumpy配列ではありません"
        assert isinstance(lower, np.ndarray), "下限バンドがnumpy配列ではありません"
        
        # 長さが元データと一致することを確認
        assert len(upper) == len(price_array), "上限バンドの長さが元データと一致しません"
        assert len(middle) == len(price_array), "中央線の長さが元データと一致しません"
        assert len(lower) == len(price_array), "下限バンドの長さが元データと一致しません"

    def test_bbands_band_relationships(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """Bollinger Bandsのバンド関係テスト"""
        price_array, _ = sample_price_data
        period, std_dev = 20, 2.0
        
        # pandas-ta版Bollinger Bands
        upper, middle, lower = pandas_ta_bbands(price_array, period, std_dev)
        
        # 有効な値のマスク
        valid_mask = ~(np.isnan(upper) | np.isnan(middle) | np.isnan(lower))
        
        if np.any(valid_mask):
            # 上限 >= 中央 >= 下限の関係を確認
            assert np.all(upper[valid_mask] >= middle[valid_mask]), "上限バンドが中央線より小さい箇所があります"
            assert np.all(middle[valid_mask] >= lower[valid_mask]), "中央線が下限バンドより小さい箇所があります"

    def test_bbands_backtesting_compatibility(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """backtesting.py互換性テスト"""
        price_array, _ = sample_price_data
        period, std_dev = 20, 2.0
        
        # pandas-ta版Bollinger Bands
        upper, middle, lower = pandas_ta_bbands(price_array, period, std_dev)
        
        # 有効な値が存在することを確認
        assert not np.isnan(upper[-10:]).all(), "上限バンドの最後の10個の値が全てNaNです"
        assert not np.isnan(middle[-10:]).all(), "中央線の最後の10個の値が全てNaNです"
        assert not np.isnan(lower[-10:]).all(), "下限バンドの最後の10個の値が全てNaNです"

    def test_bbands_error_handling(self):
        """エラーハンドリングテスト"""
        from app.services.indicators.pandas_ta_utils import PandasTAError
        
        # 空のデータ
        with pytest.raises(PandasTAError):
            pandas_ta_bbands(np.array([]), 20, 2.0)
        
        # 期間がデータ長より長い
        short_data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(PandasTAError):
            pandas_ta_bbands(short_data, 20, 2.0)
        
        # 全てNaNのデータ
        nan_data = np.array([np.nan] * 50)
        with pytest.raises(PandasTAError):
            pandas_ta_bbands(nan_data, 20, 2.0)

    def test_bbands_with_nan_values(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """NaN値を含むデータでのBollinger Bandsテスト"""
        price_array, _ = sample_price_data
        period, std_dev = 20, 2.0
        
        # 一部にNaN値を挿入
        price_with_nan = price_array.copy()
        price_with_nan[100:105] = np.nan
        
        # pandas-ta版Bollinger Bands（NaN値の処理を確認）
        upper, middle, lower = pandas_ta_bbands(price_with_nan, period, std_dev)
        
        # 結果がnumpy配列であることを確認
        assert isinstance(upper, np.ndarray)
        assert isinstance(middle, np.ndarray)
        assert isinstance(lower, np.ndarray)
        
        # NaN以外の部分で有効な値が存在することを確認
        valid_upper = upper[~np.isnan(upper)]
        assert len(valid_upper) > 0, "上限バンドに有効な値が存在しません"

    def test_bbands_middle_line_is_sma(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """Bollinger Bandsの中央線がSMAと一致することを確認"""
        price_array, _ = sample_price_data
        period, std_dev = 20, 2.0
        
        # Bollinger Bandsの中央線
        _, middle, _ = pandas_ta_bbands(price_array, period, std_dev)
        
        # SMAを直接計算
        from app.services.indicators.pandas_ta_utils import pandas_ta_sma
        sma = pandas_ta_sma(price_array, period)
        
        # 中央線とSMAが一致することを確認
        valid_mask = ~(np.isnan(middle) | np.isnan(sma))
        np.testing.assert_allclose(
            middle[valid_mask],
            sma[valid_mask],
            rtol=1e-10,
            err_msg="Bollinger Bandsの中央線がSMAと一致しません"
        )

    def test_bbands_performance_comparison(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """パフォーマンス比較テスト（参考用）"""
        import time
        
        price_array, _ = sample_price_data
        period, std_dev = 20, 2.0
        iterations = 50
        
        # TA-lib版のパフォーマンス測定
        start_time = time.time()
        for _ in range(iterations):
            VolatilityIndicators.bollinger_bands(price_array, period, std_dev)
        talib_time = time.time() - start_time
        
        # pandas-ta版のパフォーマンス測定
        start_time = time.time()
        for _ in range(iterations):
            pandas_ta_bbands(price_array, period, std_dev)
        pandas_ta_time = time.time() - start_time
        
        print(f"\nパフォーマンス比較 (Bollinger Bands, {iterations}回実行):")
        print(f"TA-lib: {talib_time:.4f}秒")
        print(f"pandas-ta: {pandas_ta_time:.4f}秒")
        print(f"比率: {pandas_ta_time/talib_time:.2f}x")

    def test_direct_talib_vs_pandas_ta_comparison(self, sample_price_data: Tuple[np.ndarray, pd.Series]):
        """直接的なTA-libとpandas-ta比較"""
        price_array, _ = sample_price_data
        period, std_dev = 20, 2.0
        
        # 直接TA-lib呼び出し
        direct_talib_upper, direct_talib_middle, direct_talib_lower = talib.BBANDS(
            price_array, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev, matype=0
        )
        
        # pandas-ta版
        pandas_ta_upper, pandas_ta_middle, pandas_ta_lower = pandas_ta_bbands(
            price_array, period, std_dev
        )
        
        # 結果の比較
        valid_mask = ~(np.isnan(direct_talib_upper) | np.isnan(pandas_ta_upper))
        np.testing.assert_allclose(
            direct_talib_upper[valid_mask],
            pandas_ta_upper[valid_mask],
            rtol=1e-10,
            err_msg="直接TA-lib呼び出しとpandas-taの上限バンド結果が一致しません"
        )


if __name__ == "__main__":
    # 基本的なテストを実行
    pytest.main([__file__, "-v"])
