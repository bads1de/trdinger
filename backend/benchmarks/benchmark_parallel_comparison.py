"""
並列評価 最適化前後比較ベンチマーク

最適化前後のパフォーマンスを直接比較するためのベンチマーク
"""

import logging
import os
import sys
import time
from typing import Any, Tuple

import numpy as np

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.services.auto_strategy.core.evaluation.parallel_evaluator import (
    ParallelEvaluator,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def simple_evaluate(individual: Any) -> Tuple[float, ...]:
    """簡易評価関数（テスト用）"""
    # 計算をシミュレート
    time.sleep(0.001)  # 1ms の計算時間
    return (np.random.random(),)


def benchmark_parallel_evaluators():
    """並列評価器のベンチマーク"""
    logger.info("=== 並列評価器 最適化前後比較 ===")

    from app.services.auto_strategy.core.evaluation.parallel_evaluator import (
        ParallelEvaluator,
    )

    OptimizedParallelEvaluator = ParallelEvaluator

    # テストデータ
    population = list(range(20))

    # オリジナル版（スレッドプール）
    original_evaluator = ParallelEvaluator(
        evaluate_func=simple_evaluate,
        max_workers=4,
        timeout_per_individual=30.0,
        use_process_pool=False,
    )

    # 最適化版
    optimized_evaluator = OptimizedParallelEvaluator(
        evaluate_func=simple_evaluate,
        max_workers=4,
        timeout_per_individual=30.0,
    )

    # ウームアップ
    original_evaluator.start()
    try:
        original_evaluator.evaluate_population(population[:5])
    finally:
        original_evaluator.shutdown()

    optimized_evaluator.start()
    try:
        optimized_evaluator.evaluate_population(population[:5])
    finally:
        optimized_evaluator.shutdown()

    # ベンチマーク実行
    n_iterations = 5

    # オリジナル版
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        original_evaluator.start()
        try:
            original_evaluator.evaluate_population(population)
        finally:
            original_evaluator.shutdown()
    original_time = time.perf_counter() - start_time

    # 最適化版
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        optimized_evaluator.start()
        try:
            optimized_evaluator.evaluate_population(population)
        finally:
            optimized_evaluator.shutdown()
    optimized_time = time.perf_counter() - start_time

    # 結果表示
    logger.info(f"\n=== 結果 ({n_iterations}回実行, 20個体) ===")
    logger.info(
        f"オリジナル版: {original_time:.4f}秒 ({original_time/n_iterations*1000:.1f}ms/回)"
    )
    logger.info(
        f"最適化版:    {optimized_time:.4f}秒 ({optimized_time/n_iterations*1000:.1f}ms/回)"
    )

    speedup = original_time / optimized_time
    logger.info(
        f"高速化率: {speedup:.2f}倍 ({(1 - optimized_time/original_time) * 100:.1f}%削減)"
    )

    return {
        "original_time": original_time,
        "optimized_time": optimized_time,
        "speedup": speedup,
    }


def benchmark_with_different_population_sizes():
    """異なる個体数でのベンチマーク"""
    logger.info("\n=== 個体数別ベンチマーク ===")

    from app.services.auto_strategy.core.evaluation.parallel_evaluator import (
        ParallelEvaluator,
    )

    OptimizedParallelEvaluator = ParallelEvaluator

    population_sizes = [10, 20, 50, 100]
    n_iterations = 3

    results = {}

    for pop_size in population_sizes:
        population = list(range(pop_size))

        # オリジナル版
        original_evaluator = ParallelEvaluator(
            evaluate_func=simple_evaluate,
            max_workers=4,
            timeout_per_individual=30.0,
            use_process_pool=False,
        )

        # 最適化版
        optimized_evaluator = OptimizedParallelEvaluator(
            evaluate_func=simple_evaluate,
            max_workers=4,
            timeout_per_individual=30.0,
        )

        # ウームアップ
        original_evaluator.start()
        try:
            original_evaluator.evaluate_population(population[:5])
        finally:
            original_evaluator.shutdown()

        optimized_evaluator.start()
        try:
            optimized_evaluator.evaluate_population(population[:5])
        finally:
            optimized_evaluator.shutdown()

        # オリジナル版
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            original_evaluator.start()
            try:
                original_evaluator.evaluate_population(population)
            finally:
                original_evaluator.shutdown()
        original_time = time.perf_counter() - start_time

        # 最適化版
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            optimized_evaluator.start()
            try:
                optimized_evaluator.evaluate_population(population)
            finally:
                optimized_evaluator.shutdown()
        optimized_time = time.perf_counter() - start_time

        speedup = original_time / optimized_time
        results[pop_size] = {
            "original_time": original_time,
            "optimized_time": optimized_time,
            "speedup": speedup,
        }

        logger.info(f"\n個体数 {pop_size}:")
        logger.info(
            f"  オリジナル: {original_time:.4f}秒 ({original_time/n_iterations*1000:.1f}ms/回)"
        )
        logger.info(
            f"  最適化版:   {optimized_time:.4f}秒 ({optimized_time/n_iterations*1000:.1f}ms/回)"
        )
        logger.info(f"  高速化率:   {speedup:.2f}倍")

    return results


def benchmark_cache_effectiveness():
    """キャッシュの効果測定"""
    logger.info("\n=== キャッシュ効果ベンチマーク ===")

    OptimizedParallelEvaluator = ParallelEvaluator

    # キャッシュあり
    evaluator_with_cache = OptimizedParallelEvaluator(
        evaluate_func=simple_evaluate,
        max_workers=4,
        timeout_per_individual=30.0,
    )

    # キャッシュなし
    evaluator_without_cache = OptimizedParallelEvaluator(
        evaluate_func=simple_evaluate,
        max_workers=4,
        timeout_per_individual=30.0,
    )

    population = list(range(20))
    n_iterations = 5

    # キャッシュあり（初回）
    evaluator_with_cache.start()
    try:
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            evaluator_with_cache.evaluate_population(population)
        time_with_cache_first = time.perf_counter() - start_time

        stats = evaluator_with_cache.get_statistics()
        logger.info(f"\nキャッシュあり（初回）:")
        logger.info(f"  時間: {time_with_cache_first:.4f}秒")
        logger.info(f"  成功率: {stats['success_rate']*100:.1f}%")
    finally:
        evaluator_with_cache.shutdown()

    # キャッシュあり（2回目 - キャッシュヒット）
    evaluator_with_cache.start()
    try:
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            evaluator_with_cache.evaluate_population(population)
        time_with_cache_second = time.perf_counter() - start_time

        stats = evaluator_with_cache.get_statistics()
        logger.info(f"\nキャッシュあり（2回目）:")
        logger.info(f"  時間: {time_with_cache_second:.4f}秒")
        logger.info(f"  成功率: {stats['success_rate']*100:.1f}%")
    finally:
        evaluator_with_cache.shutdown()

    # キャッシュなし
    evaluator_without_cache.start()
    try:
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            evaluator_without_cache.evaluate_population(population)
        time_without_cache = time.perf_counter() - start_time

        logger.info(f"\nキャッシュなし:")
        logger.info(f"  時間: {time_without_cache:.4f}秒")
    finally:
        evaluator_without_cache.shutdown()

    # 比較
    speedup_first = time_without_cache / time_with_cache_first
    speedup_second = time_without_cache / time_with_cache_second

    logger.info(f"\n=== 比較 ===")
    logger.info(f"キャッシュあり（初回） vs なし: {speedup_first:.2f}倍")
    logger.info(f"キャッシュあり（2回目） vs なし: {speedup_second:.2f}倍")

    return {
        "time_with_cache_first": time_with_cache_first,
        "time_with_cache_second": time_with_cache_second,
        "time_without_cache": time_without_cache,
        "speedup_first": speedup_first,
        "speedup_second": speedup_second,
    }


def run_all_benchmarks():
    """全ベンチマークを実行"""
    logger.info("=== 並列評価 最適化ベンチマーク開始 ===")

    all_results = {}

    try:
        all_results["basic"] = benchmark_parallel_evaluators()
    except Exception as e:
        logger.error(f"基本ベンチマークエラー: {e}")

    try:
        all_results["population_sizes"] = benchmark_with_different_population_sizes()
    except Exception as e:
        logger.error(f"個体数別ベンチマークエラー: {e}")

    try:
        all_results["cache"] = benchmark_cache_effectiveness()
    except Exception as e:
        logger.error(f"キャッシュベンチマークエラー: {e}")

    logger.info("\n=== 全ベンチマーク完了 ===")

    # サマリー表示
    if "basic" in all_results:
        logger.info(f"\n=== サマリー ===")
        logger.info(f"基本: {all_results['basic']['speedup']:.2f}倍高速化")

    return all_results


if __name__ == "__main__":
    run_all_benchmarks()
