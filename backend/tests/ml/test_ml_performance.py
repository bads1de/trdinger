"""
ML統合パフォーマンステスト

統合前後のパフォーマンス比較テスト
"""

import unittest
import time
import warnings
import psutil
import os
import pandas as pd
import numpy as np
import gc

from app.core.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
from app.core.services.auto_strategy.services.ml_indicator_service import (
    MLIndicatorService,
)


class TestMLPerformance(unittest.TestCase):
    """MLパフォーマンステスト"""

    def setUp(self):
        """テスト前の準備"""
        # 非推奨警告を無視
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # テストデータ作成（複数サイズ）
        self.small_data = self._create_test_data(100)
        self.medium_data = self._create_test_data(1000)
        self.large_data = self._create_test_data(5000)

        # プロセス情報取得
        self.process = psutil.Process(os.getpid())

    def _create_test_data(self, size: int) -> pd.DataFrame:
        """テストデータ作成"""
        np.random.seed(42)  # 再現性のため
        return pd.DataFrame(
            {
                "open": np.random.rand(size) * 100 + 50,
                "high": np.random.rand(size) * 100 + 60,
                "low": np.random.rand(size) * 100 + 40,
                "close": np.random.rand(size) * 100 + 50,
                "volume": np.random.rand(size) * 1000 + 100,
            }
        )

    def _measure_performance(self, func, *args, **kwargs):
        """パフォーマンス測定"""
        # ガベージコレクション実行
        gc.collect()

        # メモリ使用量（開始）
        memory_before = self.process.memory_info().rss / 1024 / 1024  # MB

        # 処理時間測定
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # メモリ使用量（終了）
        memory_after = self.process.memory_info().rss / 1024 / 1024  # MB

        return {
            "result": result,
            "duration_ms": (end_time - start_time) * 1000,
            "memory_before_mb": memory_before,
            "memory_after_mb": memory_after,
            "memory_diff_mb": memory_after - memory_before,
        }

    def test_small_data_performance(self):
        """小データでのパフォーマンステスト"""

        # MLOrchestrator
        orchestrator = MLOrchestrator()
        perf1 = self._measure_performance(
            orchestrator.calculate_ml_indicators, self.small_data
        )

        # MLIndicatorService
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            indicator_service = MLIndicatorService()
            perf2 = self._measure_performance(
                indicator_service.calculate_ml_indicators, self.small_data
            )

        # 結果の一致確認
        for key in perf1["result"].keys():
            np.testing.assert_array_equal(perf1["result"][key], perf2["result"][key])

        # パフォーマンス比較
        print(f"\n=== 小データ (100行) パフォーマンス ===")
        print(
            f"MLOrchestrator: {perf1['duration_ms']:.2f}ms, メモリ: {perf1['memory_diff_mb']:.2f}MB"
        )
        print(
            f"MLIndicatorService: {perf2['duration_ms']:.2f}ms, メモリ: {perf2['memory_diff_mb']:.2f}MB"
        )
        print(
            f"オーバーヘッド: {((perf2['duration_ms'] / perf1['duration_ms']) - 1) * 100:.1f}%"
        )

        # プロキシのオーバーヘッドが50%以内であることを確認（処理時間のばらつきを考慮）
        self.assertLess(perf2["duration_ms"], perf1["duration_ms"] * 1.5)

    def test_medium_data_performance(self):
        """中データでのパフォーマンステスト"""

        orchestrator = MLOrchestrator()
        perf1 = self._measure_performance(
            orchestrator.calculate_ml_indicators, self.medium_data
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            indicator_service = MLIndicatorService()
            perf2 = self._measure_performance(
                indicator_service.calculate_ml_indicators, self.medium_data
            )

        print(f"\n=== 中データ (1000行) パフォーマンス ===")
        print(
            f"MLOrchestrator: {perf1['duration_ms']:.2f}ms, メモリ: {perf1['memory_diff_mb']:.2f}MB"
        )
        print(
            f"MLIndicatorService: {perf2['duration_ms']:.2f}ms, メモリ: {perf2['memory_diff_mb']:.2f}MB"
        )
        print(
            f"オーバーヘッド: {((perf2['duration_ms'] / perf1['duration_ms']) - 1) * 100:.1f}%"
        )

        # オーバーヘッドが50%以内であることを確認（処理時間のばらつきを考慮）
        self.assertLess(perf2["duration_ms"], perf1["duration_ms"] * 1.5)

    def test_large_data_performance(self):
        """大データでのパフォーマンステスト"""

        orchestrator = MLOrchestrator()
        perf1 = self._measure_performance(
            orchestrator.calculate_ml_indicators, self.large_data
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            indicator_service = MLIndicatorService()
            perf2 = self._measure_performance(
                indicator_service.calculate_ml_indicators, self.large_data
            )

        print(f"\n=== 大データ (5000行) パフォーマンス ===")
        print(
            f"MLOrchestrator: {perf1['duration_ms']:.2f}ms, メモリ: {perf1['memory_diff_mb']:.2f}MB"
        )
        print(
            f"MLIndicatorService: {perf2['duration_ms']:.2f}ms, メモリ: {perf2['memory_diff_mb']:.2f}MB"
        )
        print(
            f"オーバーヘッド: {((perf2['duration_ms'] / perf1['duration_ms']) - 1) * 100:.1f}%"
        )

        # オーバーヘッドが50%以内であることを確認（処理時間のばらつきを考慮）
        self.assertLess(perf2["duration_ms"], perf1["duration_ms"] * 1.5)

    def test_memory_usage_comparison(self):
        """メモリ使用量比較テスト"""

        # 複数回実行してメモリ使用量を測定
        orchestrator_memory = []
        indicator_memory = []

        for i in range(5):
            # MLOrchestrator
            orchestrator = MLOrchestrator()
            perf1 = self._measure_performance(
                orchestrator.calculate_ml_indicators, self.medium_data
            )
            orchestrator_memory.append(perf1["memory_diff_mb"])
            del orchestrator

            # MLIndicatorService
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                indicator_service = MLIndicatorService()
                perf2 = self._measure_performance(
                    indicator_service.calculate_ml_indicators, self.medium_data
                )
                indicator_memory.append(perf2["memory_diff_mb"])
                del indicator_service

            gc.collect()

        avg_orchestrator = np.mean(orchestrator_memory)
        avg_indicator = np.mean(indicator_memory)

        print(f"\n=== メモリ使用量比較 (5回平均) ===")
        print(f"MLOrchestrator: {avg_orchestrator:.2f}MB")
        print(f"MLIndicatorService: {avg_indicator:.2f}MB")
        print(f"差分: {avg_indicator - avg_orchestrator:.2f}MB")

        # プロキシのメモリオーバーヘッドが1MB以内であることを確認
        self.assertLess(avg_indicator - avg_orchestrator, 1.0)

    def test_concurrent_performance(self):
        """並行処理パフォーマンステスト"""
        import threading
        import queue

        def worker(service, data, result_queue):
            """ワーカー関数"""
            start_time = time.time()
            result = service.calculate_ml_indicators(data)
            duration = (time.time() - start_time) * 1000
            result_queue.put(duration)

        # 並行実行テスト
        num_threads = 5

        # MLOrchestrator
        orchestrator_times = queue.Queue()
        threads = []

        for i in range(num_threads):
            orchestrator = MLOrchestrator()
            thread = threading.Thread(
                target=worker, args=(orchestrator, self.small_data, orchestrator_times)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # MLIndicatorService
        indicator_times = queue.Queue()
        threads = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(num_threads):
                indicator_service = MLIndicatorService()
                thread = threading.Thread(
                    target=worker,
                    args=(indicator_service, self.small_data, indicator_times),
                )
                threads.append(thread)
                thread.start()

        for thread in threads:
            thread.join()

        # 結果集計
        orchestrator_results = []
        while not orchestrator_times.empty():
            orchestrator_results.append(orchestrator_times.get())

        indicator_results = []
        while not indicator_times.empty():
            indicator_results.append(indicator_times.get())

        avg_orchestrator = np.mean(orchestrator_results)
        avg_indicator = np.mean(indicator_results)

        print(f"\n=== 並行処理パフォーマンス ({num_threads}スレッド) ===")
        print(f"MLOrchestrator: {avg_orchestrator:.2f}ms")
        print(f"MLIndicatorService: {avg_indicator:.2f}ms")
        print(f"オーバーヘッド: {((avg_indicator / avg_orchestrator) - 1) * 100:.1f}%")

        # 並行処理でもオーバーヘッドが50%以内であることを確認（処理時間のばらつきを考慮）
        self.assertLess(avg_indicator, avg_orchestrator * 1.5)

    def test_single_indicator_performance(self):
        """単一指標パフォーマンステスト"""

        indicator_types = ["ML_UP_PROB", "ML_DOWN_PROB", "ML_RANGE_PROB"]

        for indicator_type in indicator_types:
            # MLOrchestrator
            orchestrator = MLOrchestrator()
            perf1 = self._measure_performance(
                orchestrator.calculate_single_ml_indicator,
                indicator_type,
                self.medium_data,
            )

            # MLIndicatorService
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                indicator_service = MLIndicatorService()
                perf2 = self._measure_performance(
                    indicator_service.calculate_single_ml_indicator,
                    indicator_type,
                    self.medium_data,
                )

            # 結果の一致確認
            np.testing.assert_array_equal(perf1["result"], perf2["result"])

            print(f"\n=== {indicator_type} パフォーマンス ===")
            print(f"MLOrchestrator: {perf1['duration_ms']:.2f}ms")
            print(f"MLIndicatorService: {perf2['duration_ms']:.2f}ms")
            print(
                f"オーバーヘッド: {((perf2['duration_ms'] / perf1['duration_ms']) - 1) * 100:.1f}%"
            )

            # オーバーヘッドが50%以内であることを確認（処理時間のばらつきを考慮）
            self.assertLess(perf2["duration_ms"], perf1["duration_ms"] * 1.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
