"""
AutoStrategy パフォーマンスベンチマーク

最適化前後のパフォーマンスを測定するためのベンチマークスクリプト
"""

import cProfile
import logging
import os
import pstats
import sys
import time
import tracemalloc
from io import StringIO
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """ベンチマーク実行クラス"""

    def __init__(self):
        self.results: Dict[str, Dict[str, float]] = {}

    def measure_time(self, name: str, func, *args, **kwargs):
        """実行時間を測定"""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time

        if name not in self.results:
            self.results[name] = {}
        self.results[name]["time"] = elapsed

        logger.info(f"[{name}] 実行時間: {elapsed:.4f}秒")
        return result

    def measure_memory(self, name: str, func, *args, **kwargs):
        """メモリ使用量を測定"""
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if name not in self.results:
            self.results[name] = {}
        self.results[name]["memory_current"] = current / 1024 / 1024  # MB
        self.results[name]["memory_peak"] = peak / 1024 / 1024  # MB

        logger.info(
            f"[{name}] メモリ使用量: 現在={current/1024/1024:.2f}MB, ピーク={peak/1024/1024:.2f}MB"
        )
        return result

    def profile(self, name: str, func, *args, **kwargs):
        """プロファイリングを実行"""
        profiler = cProfile.Profile()
        profiler.enable()
        result = profiler.disable()

        stream = StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(20)

        logger.info(f"[{name}] プロファイル結果:\n{stream.getvalue()}")
        return result

    def print_summary(self):
        """結果のサマリーを表示"""
        logger.info("\n=== ベンチマーク結果サマリー ===")
        for name, metrics in self.results.items():
            logger.info(f"\n[{name}]")
            for metric, value in metrics.items():
                if "memory" in metric:
                    logger.info(f"  {metric}: {value:.2f} MB")
                else:
                    logger.info(f"  {metric}: {value:.4f}秒")


def create_mock_ohlcv_data(n_bars: int = 1000) -> pd.DataFrame:
    """モックOHLCVデータを生成"""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="1h")
    close = 50000 + np.cumsum(np.random.randn(n_bars) * 100)
    return pd.DataFrame(
        {
            "timestamp": dates,
            "Open": close + np.random.randn(n_bars) * 50,
            "High": close + abs(np.random.randn(n_bars) * 100),
            "Low": close - abs(np.random.randn(n_bars) * 100),
            "Close": close,
            "Volume": np.random.randint(100, 10000, n_bars),
        },
        index=dates,
    )


def create_mock_strategy_gene():
    """モック戦略遺伝子を生成"""
    from app.services.auto_strategy.genes import (
        Condition,
        IndicatorGene,
        StrategyGene,
        TPSLGene,
    )

    indicators = [
        IndicatorGene(type="SMA", period=20, source="Close"),
        IndicatorGene(type="EMA", period=50, source="Close"),
        IndicatorGene(type="RSI", period=14, source="Close"),
    ]

    long_conditions = [
        Condition(
            left_operand="sma_20", operator=">", right_operand="ema_50"
        ),
        Condition(left_operand="rsi_14", operator="<", right_operand=30),
    ]

    short_conditions = [
        Condition(
            left_operand="sma_20", operator="<", right_operand="ema_50"
        ),
        Condition(left_operand="rsi_14", operator=">", right_operand=70),
    ]

    tpsl_gene = TPSLGene(
        method="atr", sl_multiplier=2.0, tp_multiplier=3.0, atr_period=14
    )

    return StrategyGene(
        id="benchmark_gene",
        indicators=indicators,
        long_entry_conditions=long_conditions,
        short_entry_conditions=short_conditions,
        tpsl_gene=tpsl_gene,
    )


def benchmark_condition_evaluator():
    """条件評価器のベンチマーク"""
    logger.info("\n=== 条件評価器ベンチマーク ===")

    from app.services.auto_strategy.core.evaluation.condition_evaluator import (
        ConditionEvaluator,
    )
    from app.services.auto_strategy.genes import Condition

    evaluator = ConditionEvaluator()
    data = create_mock_ohlcv_data(10000)

    # モック戦略インスタンス
    class MockStrategy:
        def __init__(self, data):
            self.data = data
            self.indicators = {
                "sma_20": data["Close"].rolling(20).mean(),
                "ema_50": data["Close"].ewm(span=50).mean(),
                "rsi_14": calculate_rsi(data["Close"], 14),
            }

    def calculate_rsi(prices, period):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    strategy = MockStrategy(data)

    # 単一条件評価のベンチマーク
    condition = Condition(
        left_operand="sma_20", operator=">", right_operand="ema_50"
    )

    runner = BenchmarkRunner()

    def evaluate_single():
        for _ in range(1000):
            evaluator.evaluate_single_condition(condition, strategy)

    runner.measure_time("条件評価_1000回", evaluate_single)

    # ベクトル化評価のベンチマーク
    def evaluate_vectorized():
        for _ in range(100):
            evaluator.evaluate_single_condition_vectorized(condition, strategy)

    runner.measure_time("ベクトル化評価_100回", evaluate_vectorized)

    runner.print_summary()
    return runner.results


def benchmark_individual_evaluator():
    """個体評価器のベンチマーク"""
    logger.info("\n=== 個体評価器ベンチマーク ===")

    from unittest.mock import AsyncMock, MagicMock, patch

    from app.services.auto_strategy.config.ga import GAConfig
    from app.services.auto_strategy.core.evaluation.individual_evaluator import (
        IndividualEvaluator,
    )

    # モックバックテストサービス
    mock_backtest_service = MagicMock()
    mock_backtest_service.run_backtest.return_value = {
        "performance_metrics": {
            "total_return": 0.15,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "profit_factor": 1.8,
            "sortino_ratio": 2.0,
            "calmar_ratio": 1.5,
            "total_trades": 100,
        },
        "equity_curve": [{"drawdown": 0.05} for _ in range(100)],
        "trade_history": [{"size": 1, "pnl": 100} for _ in range(50)],
    }

    evaluator = IndividualEvaluator(mock_backtest_service, max_cache_size=100)
    config = GAConfig()
    gene = create_mock_strategy_gene()

    runner = BenchmarkRunner()

    # バックテスト設定
    backtest_config = {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-04-01",
    }
    evaluator.set_backtest_config(backtest_config)

    # 初回評価（キャッシュなし）
    def evaluate_first():
        return evaluator.evaluate_individual(gene, config)

    runner.measure_time("初回評価", evaluate_first)
    runner.measure_memory("初回評価_メモリ", evaluate_first)

    # キャッシュあり評価
    def evaluate_cached():
        return evaluator.evaluate_individual(gene, config)

    runner.measure_time("キャッシュ評価", evaluate_cached)

    # 複数遺伝子評価
    genes = [create_mock_strategy_gene() for _ in range(10)]
    for i, g in enumerate(genes):
        g.id = f"gene_{i}"

    def evaluate_multiple():
        for g in genes:
            evaluator.evaluate_individual(g, config)

    runner.measure_time("10遺伝子評価", evaluate_multiple)

    runner.print_summary()
    return runner.results


def benchmark_parallel_evaluator():
    """並列評価器のベンチマーク"""
    logger.info("\n=== 並列評価器ベンチマーク ===")

    from concurrent.futures import ProcessPoolExecutor
    from unittest.mock import MagicMock

    from app.services.auto_strategy.config.ga import GAConfig
    from app.services.auto_strategy.core.evaluation.parallel_evaluator import (
        ParallelEvaluator,
    )

    # 簡易評価関数
    def simple_evaluate(individual):
        # 計算をシミュレート
        time.sleep(0.01)  # 10ms の計算時間
        return (np.random.random(),)

    runner = BenchmarkRunner()

    # シーケンシャル評価
    individuals = list(range(20))

    def evaluate_sequential():
        return [simple_evaluate(ind) for ind in individuals]

    runner.measure_time("シーケンシャル_20個体", evaluate_sequential)

    # 並列評価（プロセスプール）
    parallel_evaluator = ParallelEvaluator(
        evaluate_func=simple_evaluate,
        max_workers=4,
        timeout_per_individual=30.0,
        use_process_pool=True,
    )

    def evaluate_parallel():
        parallel_evaluator.start()
        try:
            return parallel_evaluator.evaluate_population(individuals)
        finally:
            parallel_evaluator.shutdown()

    runner.measure_time("並列_プロセスプール_20個体", evaluate_parallel)

    # 並列評価（スレッドプール）
    thread_evaluator = ParallelEvaluator(
        evaluate_func=simple_evaluate,
        max_workers=4,
        timeout_per_individual=30.0,
        use_process_pool=False,
    )

    def evaluate_thread():
        thread_evaluator.start()
        try:
            return thread_evaluator.evaluate_population(individuals)
        finally:
            thread_evaluator.shutdown()

    runner.measure_time("並列_スレッドプール_20個体", evaluate_thread)

    runner.print_summary()
    return runner.results


def benchmark_fitness_calculator():
    """フィットネス計算のベンチマーク"""
    logger.info("\n=== フィットネス計算ベンチマーク ===")

    from app.services.auto_strategy.config.ga import GAConfig
    from app.services.auto_strategy.core.fitness.fitness_calculator import (
        FitnessCalculator,
    )

    calculator = FitnessCalculator()
    config = GAConfig()

    # モックバックテスト結果
    backtest_result = {
        "performance_metrics": {
            "total_return": 0.15,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "profit_factor": 1.8,
            "sortino_ratio": 2.0,
            "calmar_ratio": 1.5,
            "total_trades": 100,
        },
        "equity_curve": [{"drawdown": np.random.random() * 0.1} for _ in range(1000)],
        "trade_history": [
            {"size": np.random.choice([1, -1]), "pnl": np.random.random() * 100}
            for _ in range(100)
        ],
        "start_date": "2024-01-01",
        "end_date": "2024-04-01",
    }

    runner = BenchmarkRunner()

    # フィットネス計算
    def calculate_fitness():
        for _ in range(1000):
            calculator.calculate_fitness(backtest_result, config)

    runner.measure_time("フィットネス計算_1000回", calculate_fitness)

    # 多目的フィットネス計算
    config_multi = GAConfig(enable_multi_objective=True)
    config_multi.objectives = [
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
    ]

    def calculate_multi_objective():
        for _ in range(1000):
            calculator.calculate_multi_objective_fitness(
                backtest_result, config_multi
            )

    runner.measure_time("多目的フィットネス計算_1000回", calculate_multi_objective)

    runner.print_summary()
    return runner.results


def run_all_benchmarks():
    """全ベンチマークを実行"""
    logger.info("=== AutoStrategy パフォーマンスベンチマーク開始 ===")

    all_results = {}

    try:
        all_results["condition_evaluator"] = benchmark_condition_evaluator()
    except Exception as e:
        logger.error(f"条件評価器ベンチマークエラー: {e}")

    try:
        all_results["individual_evaluator"] = benchmark_individual_evaluator()
    except Exception as e:
        logger.error(f"個体評価器ベンチマークエラー: {e}")

    try:
        all_results["parallel_evaluator"] = benchmark_parallel_evaluator()
    except Exception as e:
        logger.error(f"並列評価器ベンチマークエラー: {e}")

    try:
        all_results["fitness_calculator"] = benchmark_fitness_calculator()
    except Exception as e:
        logger.error(f"フィットネス計算ベンチマークエラー: {e}")

    logger.info("\n=== 全ベンチマーク完了 ===")
    return all_results


if __name__ == "__main__":
    run_all_benchmarks()
