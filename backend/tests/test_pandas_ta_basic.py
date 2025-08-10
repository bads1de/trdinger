"""
pandas-ta 基本動作確認テスト

このテストファイルは、pandas-ta ライブラリの基本的な動作を確認し、
移行が正常に完了していることを検証します。
"""

import numpy as np
import pandas as pd
import pytest
import pandas_ta as ta
from pathlib import Path
import sys

# テスト対象のモジュールをインポート
sys.path.append(str(Path(__file__).parent.parent))

from app.services.indicators.pandas_ta_utils import (
    sma, ema, rsi, macd, atr, bbands, PandasTAError
)


class TestPandasTABasic:
    """pandas-ta 基本動作確認テストクラス"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """テストセットアップ"""
        # テスト用データの生成
        np.random.seed(42)
        n = 100
        
        # 基本的な価格データを生成
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, n)
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        self.close_data = np.array(prices[1:])
        self.high_data = self.close_data + np.random.uniform(0.5, 2.0, n)
        self.low_data = self.close_data - np.random.uniform(0.5, 2.0, n)
        self.volume_data = np.random.uniform(1000, 5000, n)

    def test_pandas_ta_import(self):
        """pandas-ta のインポートが正常に動作することを確認"""
        assert ta is not None
        assert hasattr(ta, 'sma')
        assert hasattr(ta, 'ema')
        assert hasattr(ta, 'rsi')
        assert hasattr(ta, 'macd')

    def test_pandas_ta_utils_sma(self):
        """pandas_ta_utils の SMA が正常に動作することを確認"""
        result = sma(self.close_data, 20)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.close_data)
        assert result.dtype in [np.float64, np.float32]
        
        # 最初の19個はNaNであることを確認
        assert np.all(np.isnan(result[:19]))
        # 20個目以降は有効な値であることを確認
        assert not np.all(np.isnan(result[19:]))

    def test_pandas_ta_utils_ema(self):
        """pandas_ta_utils の EMA が正常に動作することを確認"""
        result = ema(self.close_data, 20)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.close_data)
        assert result.dtype in [np.float64, np.float32]
        
        # EMAは最初から値が計算される
        assert not np.all(np.isnan(result))

    def test_pandas_ta_utils_rsi(self):
        """pandas_ta_utils の RSI が正常に動作することを確認"""
        result = rsi(self.close_data, 14)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.close_data)
        assert result.dtype in [np.float64, np.float32]
        
        # RSIは0-100の範囲内であることを確認
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= 0)
        assert np.all(valid_values <= 100)

    def test_pandas_ta_utils_macd(self):
        """pandas_ta_utils の MACD が正常に動作することを確認"""
        macd_line, signal_line, histogram = macd(self.close_data, 12, 26, 9)
        
        # 3つの配列が返されることを確認
        assert isinstance(macd_line, np.ndarray)
        assert isinstance(signal_line, np.ndarray)
        assert isinstance(histogram, np.ndarray)
        
        # 全て同じ長さであることを確認
        assert len(macd_line) == len(self.close_data)
        assert len(signal_line) == len(self.close_data)
        assert len(histogram) == len(self.close_data)
        
        # データ型の確認
        assert macd_line.dtype in [np.float64, np.float32]
        assert signal_line.dtype in [np.float64, np.float32]
        assert histogram.dtype in [np.float64, np.float32]

    def test_pandas_ta_utils_atr(self):
        """pandas_ta_utils の ATR が正常に動作することを確認"""
        result = atr(self.high_data, self.low_data, self.close_data, 14)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.close_data)
        assert result.dtype in [np.float64, np.float32]
        
        # ATRは正の値であることを確認
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= 0)

    def test_pandas_ta_utils_bbands(self):
        """pandas_ta_utils の Bollinger Bands が正常に動作することを確認"""
        upper, middle, lower = bbands(self.close_data, 20, 2.0)
        
        # 3つの配列が返されることを確認
        assert isinstance(upper, np.ndarray)
        assert isinstance(middle, np.ndarray)
        assert isinstance(lower, np.ndarray)
        
        # 全て同じ長さであることを確認
        assert len(upper) == len(self.close_data)
        assert len(middle) == len(self.close_data)
        assert len(lower) == len(self.close_data)
        
        # データ型の確認
        assert upper.dtype in [np.float64, np.float32]
        assert middle.dtype in [np.float64, np.float32]
        assert lower.dtype in [np.float64, np.float32]
        
        # バンドの順序が正しいことを確認（upper > middle > lower）
        valid_indices = ~np.isnan(upper) & ~np.isnan(middle) & ~np.isnan(lower)
        if np.any(valid_indices):
            assert np.all(upper[valid_indices] >= middle[valid_indices])
            assert np.all(middle[valid_indices] >= lower[valid_indices])

    def test_error_handling_empty_data(self):
        """空のデータでのエラーハンドリングを確認"""
        empty_data = np.array([])
        
        with pytest.raises(PandasTAError):
            sma(empty_data, 20)

    def test_error_handling_insufficient_data(self):
        """データ長が不足している場合のエラーハンドリングを確認"""
        short_data = np.array([100.0, 101.0, 102.0])
        
        with pytest.raises(PandasTAError):
            sma(short_data, 20)

    def test_error_handling_all_nan_data(self):
        """全てNaNのデータでのエラーハンドリングを確認"""
        nan_data = np.array([np.nan] * 50)
        
        with pytest.raises(PandasTAError):
            sma(nan_data, 20)

    def test_error_handling_invalid_period(self):
        """無効な期間でのエラーハンドリングを確認"""
        with pytest.raises(PandasTAError):
            sma(self.close_data, 0)
        
        with pytest.raises(PandasTAError):
            sma(self.close_data, -5)

    def test_data_type_consistency(self):
        """データ型の一貫性を確認"""
        # 整数データでのテスト
        int_data = np.array([100, 101, 102, 103, 104] * 20, dtype=np.int32)
        result = sma(int_data, 5)
        
        # 結果がfloat型であることを確認
        assert result.dtype in [np.float64, np.float32]
        assert isinstance(result, np.ndarray)

    def test_pandas_series_input(self):
        """pandas Series 入力での動作確認"""
        series_data = pd.Series(self.close_data)
        result = sma(series_data, 20)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(series_data)
        assert result.dtype in [np.float64, np.float32]

    def test_calculation_accuracy_comparison(self):
        """計算精度の比較テスト（pandas-ta直接呼び出しとの比較）"""
        # pandas-ta を直接使用
        series_data = pd.Series(self.close_data)
        direct_result = ta.sma(series_data, length=20)
        
        # pandas_ta_utils を使用
        utils_result = sma(self.close_data, 20)
        
        # 結果が一致することを確認（NaN部分を除く）
        valid_mask = ~np.isnan(direct_result.values) & ~np.isnan(utils_result)
        
        if np.any(valid_mask):
            np.testing.assert_array_almost_equal(
                direct_result.values[valid_mask],
                utils_result[valid_mask],
                decimal=10
            )

    def test_memory_efficiency(self):
        """メモリ効率のテスト"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 大量の計算を実行
        large_data = np.random.uniform(90, 110, 10000)
        for _ in range(100):
            result = sma(large_data, 20)
            del result
        
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # メモリリークがないことを確認（50MB以下の増加は許容）
        assert memory_increase < 50, f"メモリ使用量が増加しました: {memory_increase:.2f}MB"

    def test_concurrent_calculation(self):
        """並行計算のテスト"""
        import threading
        import queue
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def calculate_sma():
            try:
                result = sma(self.close_data, 20)
                results_queue.put(result)
            except Exception as e:
                errors_queue.put(str(e))
        
        # 複数スレッドで並行実行
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=calculate_sma)
            threads.append(thread)
            thread.start()
        
        # 全スレッドの完了を待機
        for thread in threads:
            thread.join()
        
        # 結果を検証
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        errors = []
        while not errors_queue.empty():
            errors.append(errors_queue.get())
        
        assert len(results) == 10, f"期待される結果数と異なります。エラー: {errors}"
        
        # 全ての結果が同じであることを確認
        for result in results[1:]:
            np.testing.assert_array_equal(results[0], result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
