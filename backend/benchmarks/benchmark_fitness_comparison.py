"""
フィットネス計算 最適化前後比較ベンチマーク

最適化前後のパフォーマンスを直接比較するためのベンチマーク
"""

import logging
import os
import sys
import time
from typing import Any, Dict

import numpy as np

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_backtest_result(n_trades: int = 100) -> Dict[str, Any]:
    """モックバックテスト結果を生成"""
    np.random.seed(42)

    # トレード履歴を生成
    trade_history = []
    for _ in range(n_trades):
        size = np.random.choice([1, -1])
        pnl = np.random.random() * 200 - 50  # -50 ~ 150の範囲
        trade_history.append({"size": size, "pnl": pnl})

    # エクイティカーブを生成
    equity_curve = []
    for _ in range(100):
        equity_curve.append({"drawdown": np.random.random() * 0.15})

    return {
        "performance_metrics": {
            "total_return": 0.15 + np.random.random() * 0.1,
            "sharpe_ratio": 1.5 + np.random.random() * 0.5,
            "max_drawdown": 0.1 + np.random.random() * 0.05,
            "win_rate": 0.6 + np.random.random() * 0.1,
            "profit_factor": 1.8 + np.random.random() * 0.2,
            "sortino_ratio": 2.0 + np.random.random() * 0.3,
            "calmar_ratio": 1.5 + np.random.random() * 0.2,
            "total_trades": n_trades,
        },
        "equity_curve": equity_curve,
        "trade_history": trade_history,
        "start_date": "2024-01-01",
        "end_date": "2024-04-01",
    }


def create_mock_ga_config():
    """モックGA設定を生成"""
    from app.services.auto_strategy.config.ga import GAConfig

    config = GAConfig()
    config.fitness_weights = {
        "total_return": 0.3,
        "sharpe_ratio": 0.4,
        "max_drawdown": 0.2,
        "win_rate": 0.1,
        "balance_score": 0.1,
        "ulcer_index_penalty": 0.05,
        "trade_frequency_penalty": 0.05,
    }
    return config


def benchmark_fitness_calculators():
    """フィットネス計算のベンチマーク"""
    logger.info("=== フィットネス計算 最適化前後比較 ===")

    from app.services.auto_strategy.core.fitness.fitness_calculator import (
        FitnessCalculator,
    )
    OptimizedFitnessCalculator = FitnessCalculator

    # 計算器の初期化
    original_calculator = FitnessCalculator()
    optimized_calculator = OptimizedFitnessCalculator()

    # テストデータ
    backtest_result = create_mock_backtest_result(100)
    config = create_mock_ga_config()

    # ウームアップ
    for _ in range(10):
        original_calculator.calculate_fitness(backtest_result, config)
        optimized_calculator.calculate_fitness(backtest_result, config)

    # ベンチマーク実行
    n_iterations = 1000

    # オリジナル版
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        original_calculator.calculate_fitness(backtest_result, config)
    original_time = time.perf_counter() - start_time

    # 最適化版
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        optimized_calculator.calculate_fitness(backtest_result, config)
    optimized_time = time.perf_counter() - start_time

    # 結果表示
    logger.info(f"\n=== 結果 ({n_iterations}回実行) ===")
    logger.info(f"オリジナル版: {original_time:.4f}秒 ({original_time/n_iterations*1000:.3f}ms/回)")
    logger.info(f"最適化版:    {optimized_time:.4f}秒 ({optimized_time/n_iterations*1000:.3f}ms/回)")

    speedup = original_time / optimized_time
    logger.info(f"高速化率: {speedup:.2f}倍 ({(1 - optimized_time/original_time) * 100:.1f}%削減)")

    # 結果の一致性確認
    original_result = original_calculator.calculate_fitness(backtest_result, config)
    optimized_result = optimized_calculator.calculate_fitness(backtest_result, config)

    logger.info(f"\n=== 結果の一致性 ===")
    logger.info(f"オリジナル: {original_result:.6f}")
    logger.info(f"最適化版:   {optimized_result:.6f}")
    logger.info(f"差分:       {abs(original_result - optimized_result):.10f}")

    return {
        "original_time": original_time,
        "optimized_time": optimized_time,
        "speedup": speedup,
        "original_result": original_result,
        "optimized_result": optimized_result,
    }


def benchmark_multi_objective_fitness():
    """多目的フィットネス計算のベンチマーク"""
    logger.info("\n=== 多目的フィットネス計算 最適化前後比較 ===")

    from app.services.auto_strategy.core.fitness.fitness_calculator import (
        FitnessCalculator,
    )
    OptimizedFitnessCalculator = FitnessCalculator

    # 計算器の初期化
    original_calculator = FitnessCalculator()
    optimized_calculator = OptimizedFitnessCalculator()

    # テストデータ
    backtest_result = create_mock_backtest_result(100)
    config = create_mock_ga_config()
    config.enable_multi_objective = True
    config.objectives = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate", "balance_score"]

    # ウームアップ
    for _ in range(10):
        original_calculator.calculate_multi_objective_fitness(backtest_result, config)
        optimized_calculator.calculate_multi_objective_fitness(backtest_result, config)

    # ベンチマーク実行
    n_iterations = 1000

    # オリジナル版
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        original_calculator.calculate_multi_objective_fitness(backtest_result, config)
    original_time = time.perf_counter() - start_time

    # 最適化版
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        optimized_calculator.calculate_multi_objective_fitness(backtest_result, config)
    optimized_time = time.perf_counter() - start_time

    # 結果表示
    logger.info(f"\n=== 結果 ({n_iterations}回実行) ===")
    logger.info(f"オリジナル版: {original_time:.4f}秒 ({original_time/n_iterations*1000:.3f}ms/回)")
    logger.info(f"最適化版:    {optimized_time:.4f}秒 ({optimized_time/n_iterations*1000:.3f}ms/回)")

    speedup = original_time / optimized_time
    logger.info(f"高速化率: {speedup:.2f}倍 ({(1 - optimized_time/original_time) * 100:.1f}%削減)")

    # 結果の一致性確認
    original_result = original_calculator.calculate_multi_objective_fitness(backtest_result, config)
    optimized_result = optimized_calculator.calculate_multi_objective_fitness(backtest_result, config)

    logger.info(f"\n=== 結果の一致性 ===")
    logger.info(f"オリジナル: {original_result}")
    logger.info(f"最適化版:   {optimized_result}")

    return {
        "original_time": original_time,
        "optimized_time": optimized_time,
        "speedup": speedup,
    }


def benchmark_with_different_trade_counts():
    """異なるトレード数でのベンチマーク"""
    logger.info("\n=== トレード数別ベンチマーク ===")

    from app.services.auto_strategy.core.fitness.fitness_calculator import (
        FitnessCalculator,
    )
    OptimizedFitnessCalculator = FitnessCalculator

    original_calculator = FitnessCalculator()
    optimized_calculator = OptimizedFitnessCalculator()
    config = create_mock_ga_config()

    trade_counts = [10, 50, 100, 500, 1000]
    n_iterations = 100

    results = {}

    for n_trades in trade_counts:
        backtest_result = create_mock_backtest_result(n_trades)

        # ウームアップ
        for _ in range(5):
            original_calculator.calculate_fitness(backtest_result, config)
            optimized_calculator.calculate_fitness(backtest_result, config)

        # オリジナル版
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            original_calculator.calculate_fitness(backtest_result, config)
        original_time = time.perf_counter() - start_time

        # 最適化版
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            optimized_calculator.calculate_fitness(backtest_result, config)
        optimized_time = time.perf_counter() - start_time

        speedup = original_time / optimized_time
        results[n_trades] = {
            "original_time": original_time,
            "optimized_time": optimized_time,
            "speedup": speedup,
        }

        logger.info(f"\nトレード数 {n_trades}:")
        logger.info(f"  オリジナル: {original_time:.4f}秒 ({original_time/n_iterations*1000:.3f}ms/回)")
        logger.info(f"  最適化版:   {optimized_time:.4f}秒 ({optimized_time/n_iterations*1000:.3f}ms/回)")
        logger.info(f"  高速化率:   {speedup:.2f}倍")

    return results


def run_all_benchmarks():
    """全ベンチマークを実行"""
    logger.info("=== フィットネス計算 最適化ベンチマーク開始 ===")

    all_results = {}

    try:
        all_results["single_objective"] = benchmark_fitness_calculators()
    except Exception as e:
        logger.error(f"単一目的ベンチマークエラー: {e}")

    try:
        all_results["multi_objective"] = benchmark_multi_objective_fitness()
    except Exception as e:
        logger.error(f"多目的ベンチマークエラー: {e}")

    try:
        all_results["trade_counts"] = benchmark_with_different_trade_counts()
    except Exception as e:
        logger.error(f"トレード数別ベンチマークエラー: {e}")

    logger.info("\n=== 全ベンチマーク完了 ===")

    # サマリー表示
    if "single_objective" in all_results:
        logger.info(f"\n=== サマリー ===")
        logger.info(f"単一目的: {all_results['single_objective']['speedup']:.2f}倍高速化")
    if "multi_objective" in all_results:
        logger.info(f"多目的: {all_results['multi_objective']['speedup']:.2f}倍高速化")

    return all_results


if __name__ == "__main__":
    run_all_benchmarks()
