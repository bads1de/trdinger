"""
バックテストパフォーマンステスト

バックテスト実行の性能とメモリ使用量をテストします。
"""

import pytest
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from backtest.engine.strategy_executor import StrategyExecutor
from app.core.strategies.sma_cross_strategy import SMACrossStrategy
from app.core.services.backtest_service import BacktestService


@pytest.mark.unit
@pytest.mark.backtest
@pytest.mark.performance
class TestBacktestPerformance:
    """バックテストパフォーマンステスト"""

    @pytest.fixture
    def large_dataset(self):
        """大きなデータセット（1年分の1時間足データ）"""
        dates = pd.date_range(
            start=datetime(2023, 1, 1, tzinfo=timezone.utc),
            end=datetime(2023, 12, 31, tzinfo=timezone.utc),
            freq="H",
        )

        np.random.seed(42)
        base_price = 50000
        prices = []

        for i in range(len(dates)):
            # リアルな価格変動をシミュレート
            change = np.random.normal(0, 0.01)  # 1%の標準偏差
            if i == 0:
                price = base_price
            else:
                price = prices[-1] * (1 + change)
            prices.append(max(price, 1000))

        return pd.DataFrame(
            {
                "Open": prices,
                "High": [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                "Low": [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                "Close": prices,
                "Volume": [1000 + np.random.randint(0, 2000) for _ in prices],
            },
            index=dates,
        )

    @pytest.fixture
    def medium_dataset(self):
        """中程度のデータセット（3ヶ月分の1時間足データ）"""
        dates = pd.date_range(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 3, 31, tzinfo=timezone.utc),
            freq="H",
        )

        np.random.seed(42)
        base_price = 50000
        prices = []

        for i in range(len(dates)):
            change = np.random.normal(0, 0.01)
            if i == 0:
                price = base_price
            else:
                price = prices[-1] * (1 + change)
            prices.append(max(price, 1000))

        return pd.DataFrame(
            {
                "Open": prices,
                "High": [p * 1.01 for p in prices],
                "Low": [p * 0.99 for p in prices],
                "Close": prices,
                "Volume": [1000] * len(prices),
            },
            index=dates,
        )

    def test_backtest_execution_time_small_dataset(self, medium_dataset):
        """小〜中規模データセットでの実行時間テスト"""
        executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)

        strategy_config = {
            "indicators": [
                {"name": "SMA", "params": {"period": 20}},
                {"name": "SMA", "params": {"period": 50}},
            ],
            "entry_rules": [{"condition": "SMA(close, 20) > SMA(close, 50)"}],
            "exit_rules": [{"condition": "SMA(close, 20) < SMA(close, 50)"}],
        }

        start_time = time.time()
        result = executor.run_backtest(medium_dataset, strategy_config)
        execution_time = time.time() - start_time

        # 実行時間が合理的な範囲内であることを確認（10秒以内）
        assert execution_time < 10.0
        assert result is not None

        print(f"中規模データセット実行時間: {execution_time:.2f}秒")
        print(f"データポイント数: {len(medium_dataset)}")

    @pytest.mark.slow
    def test_backtest_execution_time_large_dataset(self, large_dataset):
        """大規模データセットでの実行時間テスト"""
        executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)

        strategy_config = {
            "indicators": [
                {"name": "SMA", "params": {"period": 20}},
                {"name": "SMA", "params": {"period": 50}},
            ],
            "entry_rules": [{"condition": "SMA(close, 20) > SMA(close, 50)"}],
            "exit_rules": [{"condition": "SMA(close, 20) < SMA(close, 50)"}],
        }

        start_time = time.time()
        result = executor.run_backtest(large_dataset, strategy_config)
        execution_time = time.time() - start_time

        # 大規模データでも合理的な時間内で完了することを確認（30秒以内）
        assert execution_time < 30.0
        assert result is not None

        print(f"大規模データセット実行時間: {execution_time:.2f}秒")
        print(f"データポイント数: {len(large_dataset)}")

    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
    def test_memory_usage_during_backtest(self, medium_dataset):
        """バックテスト実行中のメモリ使用量テスト"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)

        strategy_config = {
            "indicators": [
                {"name": "SMA", "params": {"period": 20}},
                {"name": "SMA", "params": {"period": 50}},
            ],
            "entry_rules": [{"condition": "SMA(close, 20) > SMA(close, 50)"}],
            "exit_rules": [{"condition": "SMA(close, 20) < SMA(close, 50)"}],
        }

        result = executor.run_backtest(medium_dataset, strategy_config)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # メモリ使用量の増加が合理的な範囲内であることを確認（100MB以内）
        assert memory_increase < 100
        assert result is not None

        print(f"初期メモリ使用量: {initial_memory:.2f}MB")
        print(f"最終メモリ使用量: {final_memory:.2f}MB")
        print(f"メモリ増加量: {memory_increase:.2f}MB")

    def test_concurrent_backtest_execution(self, medium_dataset):
        """並行バックテスト実行のテスト"""
        import threading
        import queue

        def run_backtest_worker(data, config, result_queue):
            """ワーカー関数"""
            try:
                executor = StrategyExecutor(
                    initial_capital=100000, commission_rate=0.001
                )
                result = executor.run_backtest(data, config)
                result_queue.put(("success", result))
            except Exception as e:
                result_queue.put(("error", str(e)))

        strategy_config = {
            "indicators": [
                {"name": "SMA", "params": {"period": 10}},
                {"name": "SMA", "params": {"period": 20}},
            ],
            "entry_rules": [{"condition": "SMA(close, 10) > SMA(close, 20)"}],
            "exit_rules": [{"condition": "SMA(close, 10) < SMA(close, 20)"}],
        }

        # 3つの並行バックテストを実行
        num_threads = 3
        threads = []
        result_queue = queue.Queue()

        start_time = time.time()

        for i in range(num_threads):
            thread = threading.Thread(
                target=run_backtest_worker,
                args=(medium_dataset.copy(), strategy_config, result_queue),
            )
            threads.append(thread)
            thread.start()

        # 全スレッドの完了を待機
        for thread in threads:
            thread.join()

        execution_time = time.time() - start_time

        # 結果の確認
        results = []
        while not result_queue.empty():
            status, result = result_queue.get()
            results.append((status, result))

        # 全てのバックテストが成功したことを確認
        assert len(results) == num_threads
        for status, result in results:
            assert status == "success"
            assert result is not None

        print(f"並行実行時間: {execution_time:.2f}秒")
        print(f"並行実行数: {num_threads}")

    def test_backtest_service_performance(self, medium_dataset):
        """BacktestServiceのパフォーマンステスト"""
        with patch(
            "app.core.services.backtest_service.BacktestDataService"
        ) as mock_data_service:
            # モックデータサービスの設定
            mock_data_service_instance = Mock()
            mock_data_service.return_value = mock_data_service_instance
            mock_data_service_instance.get_ohlcv_for_backtest.return_value = (
                medium_dataset
            )

            service = BacktestService()
            service.data_service = mock_data_service_instance

            config = {
                "strategy_name": "SMA_CROSS",
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "end_date": datetime(2024, 3, 31, tzinfo=timezone.utc),
                "initial_capital": 100000.0,
                "commission_rate": 0.001,
                "strategy_config": {
                    "strategy_type": "SMA_CROSS",
                    "parameters": {"n1": 20, "n2": 50},
                },
            }

            start_time = time.time()
            result = service.run_backtest(config)
            execution_time = time.time() - start_time

            # サービス層での実行時間が合理的であることを確認
            assert execution_time < 15.0
            assert result is not None
            assert "performance_metrics" in result

            print(f"BacktestService実行時間: {execution_time:.2f}秒")

    def test_multiple_strategy_parameters_performance(self, medium_dataset):
        """複数の戦略パラメータでのパフォーマンステスト"""
        executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)

        # 異なるパラメータセットでテスト
        parameter_sets = [(5, 15), (10, 30), (20, 50), (30, 70)]

        results = []
        total_start_time = time.time()

        for n1, n2 in parameter_sets:
            strategy_config = {
                "indicators": [
                    {"name": "SMA", "params": {"period": n1}},
                    {"name": "SMA", "params": {"period": n2}},
                ],
                "entry_rules": [{"condition": f"SMA(close, {n1}) > SMA(close, {n2})"}],
                "exit_rules": [{"condition": f"SMA(close, {n1}) < SMA(close, {n2})"}],
            }

            start_time = time.time()
            result = executor.run_backtest(medium_dataset, strategy_config)
            execution_time = time.time() - start_time

            results.append((n1, n2, execution_time, result))

        total_execution_time = time.time() - total_start_time

        # 全ての実行が成功したことを確認
        assert len(results) == len(parameter_sets)
        for n1, n2, exec_time, result in results:
            assert result is not None
            assert exec_time < 10.0  # 各実行が10秒以内

        # 合計実行時間が合理的であることを確認
        assert total_execution_time < 40.0

        print(f"複数パラメータテスト合計時間: {total_execution_time:.2f}秒")
        print(f"パラメータセット数: {len(parameter_sets)}")

        # 実行時間の詳細を出力
        for n1, n2, exec_time, result in results:
            metrics = result["performance_metrics"]
            print(
                f"  SMA({n1},{n2}): {exec_time:.2f}秒, "
                f"リターン: {metrics['total_return']:.2f}%, "
                f"取引数: {metrics['total_trades']}"
            )


class TestBacktestScalability:
    """バックテストスケーラビリティテスト"""

    def test_data_size_scaling(self):
        """データサイズに対するスケーラビリティテスト"""
        data_sizes = [100, 500, 1000, 2000]  # データポイント数
        execution_times = []

        for size in data_sizes:
            # サイズに応じたデータを生成
            dates = pd.date_range("2024-01-01", periods=size, freq="H")
            np.random.seed(42)

            prices = [50000]
            for i in range(1, size):
                change = np.random.normal(0, 0.01)
                price = prices[-1] * (1 + change)
                prices.append(max(price, 1000))

            data = pd.DataFrame(
                {
                    "Open": prices,
                    "High": [p * 1.01 for p in prices],
                    "Low": [p * 0.99 for p in prices],
                    "Close": prices,
                    "Volume": [1000] * size,
                },
                index=dates,
            )

            executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)

            strategy_config = {
                "indicators": [
                    {"name": "SMA", "params": {"period": 20}},
                    {"name": "SMA", "params": {"period": 50}},
                ],
                "entry_rules": [{"condition": "SMA(close, 20) > SMA(close, 50)"}],
                "exit_rules": [{"condition": "SMA(close, 20) < SMA(close, 50)"}],
            }

            start_time = time.time()
            result = executor.run_backtest(data, strategy_config)
            execution_time = time.time() - start_time

            execution_times.append(execution_time)

            assert result is not None
            print(f"データサイズ {size}: {execution_time:.2f}秒")

        # 実行時間がデータサイズに対して線形またはそれ以下でスケールすることを確認
        # （厳密な線形性は要求しないが、極端な非線形性は避ける）
        for i in range(1, len(execution_times)):
            size_ratio = data_sizes[i] / data_sizes[i - 1]
            time_ratio = execution_times[i] / execution_times[i - 1]

            # 時間の増加率がデータサイズの増加率の3倍以下であることを確認
            assert time_ratio <= size_ratio * 3

        print(f"スケーラビリティテスト完了")
        print(f"データサイズ: {data_sizes}")
        print(f"実行時間: {[f'{t:.2f}s' for t in execution_times]}")
