"""
実際のDBデータを使用したバックテストパフォーマンステスト

大量データでの処理速度、メモリ使用量、スケーラビリティのテスト
"""

import pytest
import time
import psutil
import os
from datetime import datetime, timedelta
import logging
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.core.services.backtest_service import BacktestService
from app.core.services.backtest_data_service import BacktestDataService
from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository

logger = logging.getLogger(__name__)


class TestBacktestPerformanceWithRealData:
    """実際のDBデータを使用したパフォーマンステスト"""

    @pytest.fixture(scope="class")
    def db_session(self):
        """データベースセッション"""
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    @pytest.fixture(scope="class")
    def ohlcv_repo(self, db_session):
        """OHLCVリポジトリ"""
        return OHLCVRepository(db_session)

    @pytest.fixture(scope="class")
    def performance_data_info(self, ohlcv_repo):
        """パフォーマンステスト用データ情報"""
        # データが最も多いシンボルを選択
        symbols = ["BTC/USDT", "BTC/USDT:USDT", "BTCUSD"]
        timeframe = "1d"

        best_symbol = None
        max_count = 0

        for symbol in symbols:
            count = ohlcv_repo.count_records(symbol, timeframe)
            if count > max_count:
                max_count = count
                best_symbol = symbol

        if best_symbol and max_count > 100:
            latest = ohlcv_repo.get_latest_timestamp(best_symbol, timeframe)
            oldest = ohlcv_repo.get_oldest_timestamp(best_symbol, timeframe)

            return {
                "symbol": best_symbol,
                "count": max_count,
                "latest": latest,
                "oldest": oldest,
                "timeframe": timeframe,
            }

        pytest.skip("パフォーマンステストに十分なデータがありません")

    @pytest.fixture
    def backtest_service(self, db_session):
        """BacktestService with real data"""
        ohlcv_repo = OHLCVRepository(db_session)
        data_service = BacktestDataService(ohlcv_repo)
        return BacktestService(data_service)

    def measure_performance(self, func, *args, **kwargs):
        """パフォーマンス測定ヘルパー"""
        process = psutil.Process(os.getpid())

        # 開始時のメモリ使用量
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # 実行時間測定
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # 終了時のメモリ使用量
        memory_after = process.memory_info().rss / 1024 / 1024  # MB

        execution_time = end_time - start_time
        memory_used = memory_after - memory_before

        return {
            "result": result,
            "execution_time": execution_time,
            "memory_used": memory_used,
            "memory_before": memory_before,
            "memory_after": memory_after,
        }

    def test_small_dataset_performance(self, backtest_service, performance_data_info):
        """小規模データセット（30日）のパフォーマンステスト"""
        info = performance_data_info

        # 最近30日のデータを使用
        end_date = info["latest"]
        start_date = end_date - timedelta(days=30)

        config = {
            "strategy_name": "SMA_CROSS",
            "symbol": info["symbol"],
            "timeframe": "1d",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_capital": 1000000,  # BTCの高価格に対応
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 20, "n2": 50},
            },
        }

        perf = self.measure_performance(backtest_service.run_backtest, config)

        # パフォーマンス基準
        assert (
            perf["execution_time"] < 5.0
        ), f"実行時間が遅すぎます: {perf['execution_time']:.2f}秒"
        assert (
            perf["memory_used"] < 100
        ), f"メモリ使用量が多すぎます: {perf['memory_used']:.2f}MB"

        logger.info(
            f"小規模データセット - 実行時間: {perf['execution_time']:.2f}秒, "
            f"メモリ使用量: {perf['memory_used']:.2f}MB"
        )

    def test_medium_dataset_performance(self, backtest_service, performance_data_info):
        """中規模データセット（6ヶ月）のパフォーマンステスト"""
        info = performance_data_info

        # 最近6ヶ月のデータを使用
        end_date = info["latest"]
        start_date = end_date - timedelta(days=180)

        config = {
            "strategy_name": "SMA_CROSS",
            "symbol": info["symbol"],
            "timeframe": "1d",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_capital": 1000000,  # BTCの高価格に対応
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 20, "n2": 50},
            },
        }

        perf = self.measure_performance(backtest_service.run_backtest, config)

        # パフォーマンス基準（中規模データ）
        assert (
            perf["execution_time"] < 10.0
        ), f"実行時間が遅すぎます: {perf['execution_time']:.2f}秒"
        assert (
            perf["memory_used"] < 200
        ), f"メモリ使用量が多すぎます: {perf['memory_used']:.2f}MB"

        logger.info(
            f"中規模データセット - 実行時間: {perf['execution_time']:.2f}秒, "
            f"メモリ使用量: {perf['memory_used']:.2f}MB"
        )

    def test_large_dataset_performance(self, backtest_service, performance_data_info):
        """大規模データセット（2年）のパフォーマンステスト"""
        info = performance_data_info

        # 利用可能な最大期間（最大2年）を使用
        end_date = info["latest"]
        available_days = (end_date - info["oldest"]).days
        test_days = min(730, available_days - 10)  # 最大2年
        start_date = end_date - timedelta(days=test_days)

        config = {
            "strategy_name": "SMA_CROSS",
            "symbol": info["symbol"],
            "timeframe": "1d",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_capital": 1000000,  # BTCの高価格に対応
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 20, "n2": 50},
            },
        }

        perf = self.measure_performance(backtest_service.run_backtest, config)

        # パフォーマンス基準（大規模データ）
        assert (
            perf["execution_time"] < 30.0
        ), f"実行時間が遅すぎます: {perf['execution_time']:.2f}秒"
        assert (
            perf["memory_used"] < 500
        ), f"メモリ使用量が多すぎます: {perf['memory_used']:.2f}MB"

        logger.info(
            f"大規模データセット ({test_days}日間) - 実行時間: {perf['execution_time']:.2f}秒, "
            f"メモリ使用量: {perf['memory_used']:.2f}MB"
        )

    def test_optimization_performance(self, backtest_service, performance_data_info):
        """最適化機能のパフォーマンステスト"""
        info = performance_data_info

        # 最近3ヶ月のデータを使用（最適化は計算量が多いため）
        end_date = info["latest"]
        start_date = end_date - timedelta(days=90)

        config = {
            "strategy_name": "SMA_CROSS",
            "symbol": info["symbol"],
            "timeframe": "1d",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_capital": 1000000,  # BTCの高価格に対応
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 20, "n2": 50},
            },
        }

        # 最適化パラメータ（小規模）
        optimization_params = {
            "parameters": {
                "n1": [10, 15, 20],  # 3つの値
                "n2": [30, 40, 50],  # 3つの値 = 9通りの組み合わせ
            },
            "maximize": "Sharpe Ratio",
            "constraint": lambda p: p.n1 < p.n2,
        }

        perf = self.measure_performance(
            backtest_service.optimize_strategy, config, optimization_params
        )

        # 最適化のパフォーマンス基準
        assert (
            perf["execution_time"] < 60.0
        ), f"最適化実行時間が遅すぎます: {perf['execution_time']:.2f}秒"
        assert (
            perf["memory_used"] < 300
        ), f"最適化メモリ使用量が多すぎます: {perf['memory_used']:.2f}MB"

        logger.info(
            f"最適化パフォーマンス - 実行時間: {perf['execution_time']:.2f}秒, "
            f"メモリ使用量: {perf['memory_used']:.2f}MB"
        )

    def test_concurrent_backtest_performance(
        self, backtest_service, performance_data_info
    ):
        """並行バックテストのパフォーマンステスト"""
        info = performance_data_info

        # 複数の異なる期間でバックテストを実行
        end_date = info["latest"]
        test_configs = []

        for i in range(3):  # 3つの異なる期間
            start_date = end_date - timedelta(days=60 + i * 30)
            config = {
                "strategy_name": "SMA_CROSS",
                "symbol": info["symbol"],
                "timeframe": "1d",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "initial_capital": 1000000,  # BTCの高価格に対応
                "commission_rate": 0.001,
                "strategy_config": {
                    "strategy_type": "SMA_CROSS",
                    "parameters": {
                        "n1": 10 + i * 5,  # 異なるパラメータ
                        "n2": 30 + i * 10,
                    },
                },
            }
            test_configs.append(config)

        def run_multiple_backtests():
            results = []
            for config in test_configs:
                result = backtest_service.run_backtest(config)
                results.append(result)
            return results

        perf = self.measure_performance(run_multiple_backtests)

        # 並行処理のパフォーマンス基準
        assert (
            perf["execution_time"] < 20.0
        ), f"並行実行時間が遅すぎます: {perf['execution_time']:.2f}秒"
        assert (
            perf["memory_used"] < 400
        ), f"並行実行メモリ使用量が多すぎます: {perf['memory_used']:.2f}MB"

        # 結果の検証
        results = perf["result"]
        assert len(results) == 3, "全てのバックテストが完了していません"

        for i, result in enumerate(results):
            assert (
                "performance_metrics" in result
            ), f"結果{i+1}にパフォーマンス指標がありません"

        logger.info(
            f"並行バックテスト - 実行時間: {perf['execution_time']:.2f}秒, "
            f"メモリ使用量: {perf['memory_used']:.2f}MB"
        )

    def test_memory_leak_detection(self, backtest_service, performance_data_info):
        """メモリリーク検出テスト"""
        info = performance_data_info

        # 同じバックテストを複数回実行してメモリ使用量を監視
        end_date = info["latest"]
        start_date = end_date - timedelta(days=30)

        config = {
            "strategy_name": "SMA_CROSS",
            "symbol": info["symbol"],
            "timeframe": "1d",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_capital": 1000000,  # BTCの高価格に対応
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 20, "n2": 50},
            },
        }

        process = psutil.Process(os.getpid())
        memory_usage = []

        # 5回実行してメモリ使用量を記録
        for i in range(5):
            backtest_service.run_backtest(config)
            memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB

        # メモリ使用量の増加を確認
        memory_increase = memory_usage[-1] - memory_usage[0]

        # メモリリークの検出（50MB以上の増加は異常）
        assert (
            memory_increase < 50
        ), f"メモリリークの可能性: {memory_increase:.2f}MB増加"

        logger.info(
            f"メモリリークテスト - 初期: {memory_usage[0]:.2f}MB, "
            f"最終: {memory_usage[-1]:.2f}MB, 増加: {memory_increase:.2f}MB"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
