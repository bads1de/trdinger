"""
pandas-ta互換性テスト

型変換なし実装でのpandas-taライブラリとの互換性を確認するテストです。
"""

import pytest
import pandas as pd
import numpy as np
import pandas_ta as ta

from app.services.indicators.technical_indicators.trend import TrendIndicators
from app.services.indicators.technical_indicators.momentum import MomentumIndicators
from app.services.indicators.technical_indicators.volatility import VolatilityIndicators
from app.services.indicators.technical_indicators.volume import VolumeIndicators


class TestPandasTACompatibility:
    """pandas-ta互換性テストクラス"""

    @pytest.fixture
    def sample_ohlcv(self):
        """OHLCV形式のサンプルデータ"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="1D")

        # 現実的な価格データ生成
        base_price = 100
        returns = np.random.normal(0, 0.02, 100)
        prices = [base_price]

        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 10))

        closes = np.array(prices)
        highs = closes * (1 + np.abs(np.random.normal(0, 0.01, 100)))
        lows = closes * (1 - np.abs(np.random.normal(0, 0.01, 100)))
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        volumes = np.random.randint(1000, 10000, 100)

        return pd.DataFrame(
            {
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": closes,
                "Volume": volumes,
            },
            index=dates,
        )

    def test_sma_compatibility(self, sample_ohlcv):
        """SMAのpandas-ta互換性テスト"""
        close_series = sample_ohlcv["Close"]
        period = 20

        # 我々の実装
        our_result = TrendIndicators.sma(close_series, period)

        # pandas-taの直接実行
        pandas_ta_result = ta.sma(close_series, length=period)

        # 結果の比較（NaN部分を除く）
        valid_mask = ~(np.isnan(our_result) | pandas_ta_result.isna())
        np.testing.assert_array_almost_equal(
            our_result[valid_mask], pandas_ta_result.values[valid_mask], decimal=10
        )

    def test_ema_compatibility(self, sample_ohlcv):
        """EMAのpandas-ta互換性テスト"""
        close_series = sample_ohlcv["Close"]
        period = 20

        # 我々の実装
        our_result = TrendIndicators.ema(close_series, period)

        # pandas-taの直接実行
        pandas_ta_result = ta.ema(close_series, length=period)

        # 結果の比較
        valid_mask = ~(np.isnan(our_result) | pandas_ta_result.isna())
        np.testing.assert_array_almost_equal(
            our_result[valid_mask],
            pandas_ta_result.values[valid_mask],
            decimal=8,  # EMAは計算誤差が若干大きい
        )

    def test_rsi_compatibility(self, sample_ohlcv):
        """RSIのpandas-ta互換性テスト"""
        close_series = sample_ohlcv["Close"]
        period = 14

        # 我々の実装
        our_result = MomentumIndicators.rsi(close_series, period)

        # pandas-taの直接実行
        pandas_ta_result = ta.rsi(close_series, length=period)

        # 結果の比較
        valid_mask = ~(np.isnan(our_result) | pandas_ta_result.isna())
        np.testing.assert_array_almost_equal(
            our_result[valid_mask], pandas_ta_result.values[valid_mask], decimal=8
        )

    def test_macd_compatibility(self, sample_ohlcv):
        """MACDのpandas-ta互換性テスト"""
        close_series = sample_ohlcv["Close"]
        fast, slow, signal = 12, 26, 9

        # 我々の実装
        our_macd, our_signal, our_histogram = MomentumIndicators.macd(
            close_series, fast, slow, signal
        )

        # pandas-taの直接実行
        pandas_ta_result = ta.macd(close_series, fast=fast, slow=slow, signal=signal)

        # 結果の比較（列名を動的に取得）
        columns = pandas_ta_result.columns.tolist()
        macd_col = [
            col
            for col in columns
            if "MACD_" in col and "MACDs" not in col and "MACDh" not in col
        ][0]
        signal_col = [col for col in columns if "MACDs" in col][0]
        histogram_col = [col for col in columns if "MACDh" in col][0]

        for our_data, ta_column in [
            (our_macd, macd_col),
            (our_signal, signal_col),
            (our_histogram, histogram_col),
        ]:
            valid_mask = ~(np.isnan(our_data) | pandas_ta_result[ta_column].isna())
            np.testing.assert_array_almost_equal(
                our_data[valid_mask],
                pandas_ta_result[ta_column].values[valid_mask],
                decimal=8,
            )

    def test_bollinger_bands_compatibility(self, sample_ohlcv):
        """ボリンジャーバンドのpandas-ta互換性テスト"""
        close_series = sample_ohlcv["Close"]
        period, std = 20, 2

        # 我々の実装
        our_upper, our_middle, our_lower = VolatilityIndicators.bbands(
            close_series, period, std
        )

        # pandas-taの直接実行
        pandas_ta_result = ta.bbands(close_series, length=period, std=std)

        # 結果の比較（列名を動的に取得）
        columns = pandas_ta_result.columns.tolist()
        upper_col = [col for col in columns if "BBU" in col][0]
        middle_col = [col for col in columns if "BBM" in col][0]
        lower_col = [col for col in columns if "BBL" in col][0]

        for our_data, ta_column in [
            (our_upper, upper_col),
            (our_middle, middle_col),
            (our_lower, lower_col),
        ]:
            valid_mask = ~(np.isnan(our_data) | pandas_ta_result[ta_column].isna())
            np.testing.assert_array_almost_equal(
                our_data[valid_mask],
                pandas_ta_result[ta_column].values[valid_mask],
                decimal=10,
            )

    def test_atr_compatibility(self, sample_ohlcv):
        """ATRのpandas-ta互換性テスト"""
        high_series = sample_ohlcv["High"]
        low_series = sample_ohlcv["Low"]
        close_series = sample_ohlcv["Close"]
        period = 14

        # 我々の実装
        our_result = VolatilityIndicators.atr(
            high_series, low_series, close_series, period
        )

        # pandas-taの直接実行
        pandas_ta_result = ta.atr(high_series, low_series, close_series, length=period)

        # 結果の比較
        valid_mask = ~(np.isnan(our_result) | pandas_ta_result.isna())
        np.testing.assert_array_almost_equal(
            our_result[valid_mask], pandas_ta_result.values[valid_mask], decimal=10
        )

    def test_obv_compatibility(self, sample_ohlcv):
        """OBVのpandas-ta互換性テスト"""
        close_series = sample_ohlcv["Close"]
        volume_series = sample_ohlcv["Volume"]

        # 我々の実装
        our_result = VolumeIndicators.obv(close_series, volume_series)

        # pandas-taの直接実行
        pandas_ta_result = ta.obv(close_series, volume_series)

        # 結果の比較
        valid_mask = ~(np.isnan(our_result) | pandas_ta_result.isna())
        np.testing.assert_array_almost_equal(
            our_result[valid_mask], pandas_ta_result.values[valid_mask], decimal=10
        )

    def test_stochastic_compatibility(self, sample_ohlcv):
        """ストキャスティクスのpandas-ta互換性テスト"""
        high_series = sample_ohlcv["High"]
        low_series = sample_ohlcv["Low"]
        close_series = sample_ohlcv["Close"]
        k_period, d_period = 14, 3

        # 我々の実装
        our_k, our_d = MomentumIndicators.stoch(
            high_series, low_series, close_series, k_period, d_period
        )

        # pandas-taの直接実行
        pandas_ta_result = ta.stoch(
            high_series, low_series, close_series, k=k_period, d=d_period
        )
        # インデックスを入力に合わせて揃える（pandas-taのstochは先頭をドロップするため短くなる）
        pandas_ta_result = pandas_ta_result.reindex(close_series.index)

        # 結果の比較（列名を動的に取得）
        columns = pandas_ta_result.columns.tolist()
        k_col = [col for col in columns if "STOCHk" in col][0]
        d_col = [col for col in columns if "STOCHd" in col][0]

        for our_data, ta_column in [
            (our_k, k_col),
            (our_d, d_col),
        ]:
            valid_mask = ~(np.isnan(our_data) | pandas_ta_result[ta_column].isna())
            np.testing.assert_array_almost_equal(
                our_data[valid_mask],
                pandas_ta_result[ta_column].values[valid_mask],
                decimal=8,
            )

    def test_series_vs_array_input(self, sample_ohlcv):
        """Series入力とArray入力の結果一致テスト"""
        close_series = sample_ohlcv["Close"]
        close_array = close_series.to_numpy()
        period = 20

        # Series入力
        series_result = TrendIndicators.sma(close_series, period)

        # Array入力
        array_result = TrendIndicators.sma(close_array, period)

        # 結果が同じであることを確認
        np.testing.assert_array_equal(series_result, array_result)

    def test_data_type_preservation(self, sample_ohlcv):
        """データ型保持のテスト"""
        close_series = sample_ohlcv["Close"]

        # float32入力
        close_float32 = close_series.astype(np.float32)
        result_float32 = TrendIndicators.sma(close_float32, 20)

        # float64入力
        close_float64 = close_series.astype(np.float64)
        result_float64 = TrendIndicators.sma(close_float64, 20)

        # 両方ともfloat64で出力されることを確認
        assert result_float32.dtype == np.float64
        assert result_float64.dtype == np.float64

        # 値は同じであることを確認
        np.testing.assert_array_almost_equal(result_float32, result_float64, decimal=6)

    def test_index_preservation(self, sample_ohlcv):
        """インデックス保持のテスト"""
        close_series = sample_ohlcv["Close"]

        # インデックス付きSeries
        indexed_series = pd.Series(close_series.values, index=close_series.index)
        result = TrendIndicators.sma(indexed_series, 20)

        # 結果がnumpy配列であることを確認（インデックスは保持されない）
        assert isinstance(result, np.ndarray)
        assert len(result) == len(indexed_series)

    def test_nan_handling(self, sample_ohlcv):
        """NaN値の処理テスト"""
        close_series = sample_ohlcv["Close"].copy()

        # いくつかの値をNaNに設定
        close_series.iloc[10:15] = np.nan

        # 我々の実装
        our_result = TrendIndicators.sma(close_series, 20)

        # pandas-taの直接実行
        pandas_ta_result = ta.sma(close_series, length=20)

        # NaN部分の処理が同じであることを確認
        our_nan_mask = np.isnan(our_result)
        ta_nan_mask = pandas_ta_result.isna()

        # 有効な部分での値の比較
        valid_mask = ~(our_nan_mask | ta_nan_mask)
        if valid_mask.any():
            np.testing.assert_array_almost_equal(
                our_result[valid_mask], pandas_ta_result.values[valid_mask], decimal=10
            )

    def test_edge_case_small_dataset(self):
        """小さなデータセットでのエッジケーステスト"""
        # 最小限のデータ
        small_data = pd.Series([100, 101, 102, 103, 104])

        # 期間がデータ長と同じ
        result = TrendIndicators.sma(small_data, 5)
        assert isinstance(result, np.ndarray)
        assert len(result) == 5

        # 期間がデータ長より短い
        result = TrendIndicators.sma(small_data, 3)
        assert isinstance(result, np.ndarray)
        assert len(result) == 5

    def test_performance_comparison(self, sample_ohlcv):
        """パフォーマンス比較テスト"""
        import time

        close_series = sample_ohlcv["Close"]
        period = 20
        iterations = 100

        # 我々の実装のパフォーマンス
        start_time = time.time()
        for _ in range(iterations):
            TrendIndicators.sma(close_series, period)
        our_time = time.time() - start_time

        # pandas-taの直接実行のパフォーマンス
        start_time = time.time()
        for _ in range(iterations):
            ta.sma(close_series, length=period)
        pandas_ta_time = time.time() - start_time

        # 我々の実装がpandas-taの10倍以上遅くないことを確認
        assert (
            our_time < pandas_ta_time * 10
        ), f"Our implementation too slow: {our_time:.4f}s vs {pandas_ta_time:.4f}s"
