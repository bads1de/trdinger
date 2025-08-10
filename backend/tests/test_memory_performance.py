"""
メモリとパフォーマンスのテスト

型変換なし実装でのメモリ使用量とパフォーマンスを確認するテストです。
"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
from typing import List, Tuple

from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
from app.services.indicators.technical_indicators.trend import TrendIndicators
from app.services.indicators.technical_indicators.momentum import MomentumIndicators
from app.services.auto_strategy.calculators.indicator_calculator import (
    IndicatorCalculator,
)


class TestMemoryPerformance:
    """メモリとパフォーマンステストクラス"""

    @pytest.fixture
    def large_dataset(self):
        """大規模データセット（1年分の1分足データ）"""
        np.random.seed(42)
        size = 525600  # 1年分の分足データ

        # メモリ効率的なデータ生成
        base_price = 50000
        returns = np.random.normal(0, 0.001, size)

        # 累積リターンから価格を計算
        cumulative_returns = np.cumprod(1 + returns)
        closes = base_price * cumulative_returns

        # OHLC生成
        noise = np.random.normal(0, 0.0005, size)
        highs = closes * (1 + np.abs(noise))
        lows = closes * (1 - np.abs(noise))
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        volumes = np.random.randint(100, 1000, size)

        return pd.DataFrame(
            {
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": closes,
                "Volume": volumes,
            }
        )

    @pytest.fixture
    def medium_dataset(self):
        """中規模データセット（1ヶ月分の1分足データ）"""
        np.random.seed(42)
        size = 43200  # 1ヶ月分の分足データ

        base_price = 50000
        returns = np.random.normal(0, 0.001, size)
        cumulative_returns = np.cumprod(1 + returns)
        closes = base_price * cumulative_returns

        noise = np.random.normal(0, 0.0005, size)
        highs = closes * (1 + np.abs(noise))
        lows = closes * (1 - np.abs(noise))
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        volumes = np.random.randint(100, 1000, size)

        return pd.DataFrame(
            {
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": closes,
                "Volume": volumes,
            }
        )

    def get_memory_usage(self) -> float:
        """現在のメモリ使用量を取得（MB）"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def test_memory_usage_single_indicator(self, large_dataset):
        """単一指標でのメモリ使用量テスト"""
        initial_memory = self.get_memory_usage()

        service = TechnicalIndicatorService()

        # SMA計算
        result = service.calculate_indicator(large_dataset, "SMA", {"period": 20})

        peak_memory = self.get_memory_usage()
        memory_increase = peak_memory - initial_memory

        # メモリ増加が500MB以下であることを確認
        assert (
            memory_increase < 500
        ), f"Memory increase too large: {memory_increase:.2f} MB"

        # 結果の妥当性確認
        assert isinstance(result, np.ndarray)
        assert len(result) == len(large_dataset)

    def test_memory_usage_multiple_indicators(self, medium_dataset):
        """複数指標でのメモリ使用量テスト"""
        initial_memory = self.get_memory_usage()

        service = TechnicalIndicatorService()

        # 複数の指標を順次計算
        indicators = [
            ("SMA", {"period": 20}),
            ("EMA", {"period": 20}),
            ("RSI", {"period": 14}),
            ("MACD", {"fast": 12, "slow": 26, "signal": 9}),
            ("BB", {"period": 20, "std": 2}),
            ("ATR", {"period": 14}),
        ]

        results = []
        for indicator_name, params in indicators:
            result = service.calculate_indicator(medium_dataset, indicator_name, params)
            results.append(result)

        peak_memory = self.get_memory_usage()
        memory_increase = peak_memory - initial_memory

        # メモリ増加が200MB以下であることを確認
        assert (
            memory_increase < 200
        ), f"Memory increase too large: {memory_increase:.2f} MB"

        # 全ての結果が有効であることを確認
        for i, result in enumerate(results):
            if isinstance(result, tuple):
                for output in result:
                    assert len(output) == len(medium_dataset)
            else:
                assert len(result) == len(medium_dataset)

    def test_memory_leak_detection(self, medium_dataset):
        """メモリリーク検出テスト"""
        service = TechnicalIndicatorService()

        memory_readings = []

        # 100回の計算を実行してメモリ使用量を記録
        for i in range(100):
            service.calculate_indicator(medium_dataset, "SMA", {"period": 20})

            if i % 10 == 0:  # 10回ごとにメモリ使用量を記録
                memory_readings.append(self.get_memory_usage())

        # メモリ使用量の増加傾向を確認
        if len(memory_readings) > 2:
            # 最初と最後のメモリ使用量の差
            memory_growth = memory_readings[-1] - memory_readings[0]

            # メモリ増加が50MB以下であることを確認
            assert (
                memory_growth < 50
            ), f"Potential memory leak detected: {memory_growth:.2f} MB growth"

    def test_performance_single_indicator(self, medium_dataset):
        """単一指標のパフォーマンステスト"""
        service = TechnicalIndicatorService()

        # ウォームアップ
        service.calculate_indicator(medium_dataset, "SMA", {"period": 20})

        # パフォーマンス測定
        start_time = time.time()
        iterations = 10

        for _ in range(iterations):
            result = service.calculate_indicator(medium_dataset, "SMA", {"period": 20})

        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / iterations

        # 平均処理時間が1秒以下であることを確認
        assert (
            avg_time < 1.0
        ), f"Performance too slow: {avg_time:.3f} seconds per calculation"

    def test_performance_multiple_indicators(self, medium_dataset):
        """複数指標のパフォーマンステスト"""
        service = TechnicalIndicatorService()

        indicators = [
            ("SMA", {"period": 20}),
            ("EMA", {"period": 20}),
            ("RSI", {"period": 14}),
            ("MACD", {"fast": 12, "slow": 26, "signal": 9}),
            ("ATR", {"period": 14}),
        ]

        # ウォームアップ
        for indicator_name, params in indicators:
            service.calculate_indicator(medium_dataset, indicator_name, params)

        # パフォーマンス測定
        start_time = time.time()
        iterations = 5

        for _ in range(iterations):
            for indicator_name, params in indicators:
                service.calculate_indicator(medium_dataset, indicator_name, params)

        elapsed_time = time.time() - start_time
        avg_time_per_indicator = elapsed_time / (iterations * len(indicators))

        # 指標あたりの平均処理時間が0.5秒以下であることを確認
        assert (
            avg_time_per_indicator < 0.5
        ), f"Performance too slow: {avg_time_per_indicator:.3f} seconds per indicator"

    def test_series_vs_array_performance(self, medium_dataset):
        """Series vs Array入力のパフォーマンス比較"""
        close_series = medium_dataset["Close"]
        close_array = close_series.to_numpy()

        iterations = 20

        # pandas.Seriesでのパフォーマンス
        start_time = time.time()
        for _ in range(iterations):
            TrendIndicators.sma(close_series, 20)
        series_time = time.time() - start_time

        # numpy配列でのパフォーマンス
        start_time = time.time()
        for _ in range(iterations):
            TrendIndicators.sma(close_array, 20)
        array_time = time.time() - start_time

        # 性能差が5倍以内であることを確認
        ratio = max(series_time, array_time) / min(series_time, array_time)
        assert (
            ratio < 5.0
        ), f"Performance difference too large: Series={series_time:.3f}s, Array={array_time:.3f}s"

    def test_concurrent_performance(self, medium_dataset):
        """並行処理でのパフォーマンステスト"""
        import threading
        import queue

        service = TechnicalIndicatorService()
        results_queue = queue.Queue()
        errors_queue = queue.Queue()

        def calculate_indicator():
            try:
                start_time = time.time()
                result = service.calculate_indicator(
                    medium_dataset, "SMA", {"period": 20}
                )
                elapsed_time = time.time() - start_time
                results_queue.put(elapsed_time)
            except Exception as e:
                errors_queue.put(e)

        # 5つのスレッドで同時実行
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=calculate_indicator)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # エラーが発生していないことを確認
        assert (
            errors_queue.empty()
        ), f"Concurrent execution errors: {list(errors_queue.queue)}"

        # 全てのスレッドが完了していることを確認
        assert results_queue.qsize() == 5, "Not all threads completed"

        # 処理時間が合理的であることを確認
        times = []
        while not results_queue.empty():
            times.append(results_queue.get())

        avg_time = sum(times) / len(times)
        assert (
            avg_time < 2.0
        ), f"Concurrent performance too slow: {avg_time:.3f} seconds average"

    def test_auto_strategy_performance(self, medium_dataset):
        """オートストラテジーでのパフォーマンステスト"""

        class MockData:
            def __init__(self, df):
                self.df = df

        mock_data = MockData(medium_dataset)
        calculator = IndicatorCalculator()

        indicators = [
            ("SMA", {"period": 20}),
            ("RSI", {"period": 14}),
            ("MACD", {"fast": 12, "slow": 26, "signal": 9}),
        ]

        # パフォーマンス測定
        start_time = time.time()
        iterations = 10

        for _ in range(iterations):
            for indicator_name, params in indicators:
                calculator.calculate_indicator(mock_data, indicator_name, params)

        elapsed_time = time.time() - start_time
        avg_time_per_calculation = elapsed_time / (iterations * len(indicators))

        # 計算あたりの平均処理時間が0.3秒以下であることを確認
        assert (
            avg_time_per_calculation < 0.3
        ), f"Auto strategy performance too slow: {avg_time_per_calculation:.3f} seconds"

    def test_garbage_collection_efficiency(self, medium_dataset):
        """ガベージコレクション効率性テスト"""
        import gc

        service = TechnicalIndicatorService()

        # 初期メモリ使用量
        gc.collect()
        initial_memory = self.get_memory_usage()

        # 大量の計算を実行
        for i in range(50):
            result = service.calculate_indicator(medium_dataset, "SMA", {"period": 20})

            # 定期的にガベージコレクションを実行
            if i % 10 == 0:
                gc.collect()

        # 最終メモリ使用量
        gc.collect()
        final_memory = self.get_memory_usage()

        memory_increase = final_memory - initial_memory

        # メモリ増加が100MB以下であることを確認
        assert (
            memory_increase < 100
        ), f"Memory not properly released: {memory_increase:.2f} MB increase"

    def test_data_size_scalability(self):
        """データサイズスケーラビリティテスト"""
        service = TechnicalIndicatorService()

        sizes = [1000, 5000, 10000, 50000]
        times = []

        for size in sizes:
            # サイズに応じたデータ生成
            data = pd.DataFrame(
                {
                    "Open": np.random.rand(size) * 100 + 50000,
                    "High": np.random.rand(size) * 100 + 50100,
                    "Low": np.random.rand(size) * 100 + 49900,
                    "Close": np.random.rand(size) * 100 + 50000,
                    "Volume": np.random.randint(100, 1000, size),
                }
            )

            # パフォーマンス測定
            start_time = time.perf_counter()
            service.calculate_indicator(data, "SMA", {"period": 20})
            elapsed_time = time.perf_counter() - start_time

            times.append(elapsed_time)

        # 処理時間がデータサイズに対して線形に近いスケーリングであることを確認
        # 最大データサイズでも5秒以内で処理できることを確認
        assert (
            times[-1] < 5.0
        ), f"Scalability issue: {times[-1]:.3f} seconds for {sizes[-1]} records"

        # 処理時間の増加率が合理的であることを確認（ゼロ除算保護）
        time_ratios = [
            times[i] / (times[i - 1] if times[i - 1] > 0 else 1e-9)
            for i in range(1, len(times))
        ]
        size_ratios = [sizes[i] / sizes[i - 1] for i in range(1, len(sizes))]

        for time_ratio, size_ratio in zip(time_ratios, size_ratios):
            # 処理時間の増加率がデータサイズの増加率の2倍以下であることを確認
            assert (
                time_ratio <= size_ratio * 2
            ), f"Non-linear scaling detected: time ratio {time_ratio:.2f}, size ratio {size_ratio:.2f}"
