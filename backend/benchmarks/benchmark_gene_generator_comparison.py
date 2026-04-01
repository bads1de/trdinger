"""
遺伝子生成器 最適化前後比較ベンチマーク
"""

import logging
import os
import sys
import time

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_config():
    """モックGA設定を生成"""
    from app.services.auto_strategy.config.ga import GAConfig

    config = GAConfig()
    config.max_indicators = 5
    config.min_indicators = 1
    config.max_conditions = 3
    config.min_conditions = 1
    return config


def benchmark_gene_generators():
    """遺伝子生成器のベンチマーク"""
    logger.info("=== 遺伝子生成器 最適化前後比較 ===")

    from app.services.auto_strategy.generators.random_gene_generator import (
        RandomGeneGenerator,
    )
    OptimizedGeneGenerator = RandomGeneGenerator

    config = create_mock_config()

    # オリジナル版
    original_generator = RandomGeneGenerator(config)

    # 最適化版
    optimized_generator = OptimizedGeneGenerator(config)

    # ウームアップ
    for _ in range(5):
        original_generator.generate_random_gene()
        optimized_generator.generate_random_gene()

    # ベンチマーク実行
    n_iterations = 100

    # オリジナル版
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        original_generator.generate_random_gene()
    original_time = time.perf_counter() - start_time

    # 最適化版（初回 - キャッシュ構築）
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        optimized_generator.generate_random_gene()
    optimized_time_first = time.perf_counter() - start_time

    # 最適化版（2回目 - キャッシュ利用）
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        optimized_generator.generate_random_gene()
    optimized_time_second = time.perf_counter() - start_time

    # 結果表示
    logger.info(f"\n=== 結果 ({n_iterations}回実行) ===")
    logger.info(f"オリジナル版: {original_time:.4f}秒 ({original_time/n_iterations*1000:.3f}ms/回)")
    logger.info(f"最適化版（初回）: {optimized_time_first:.4f}秒 ({optimized_time_first/n_iterations*1000:.3f}ms/回)")
    logger.info(f"最適化版（2回目）: {optimized_time_second:.4f}秒 ({optimized_time_second/n_iterations*1000:.3f}ms/回)")

    speedup_first = original_time / optimized_time_first
    speedup_second = original_time / optimized_time_second

    logger.info(f"\n=== 高速化率 ===")
    logger.info(f"最適化版（初回）: {speedup_first:.2f}倍 ({(1 - optimized_time_first/original_time) * 100:.1f}%削減)")
    logger.info(f"最適化版（2回目）: {speedup_second:.2f}倍 ({(1 - optimized_time_second/original_time) * 100:.1f}%削減)")

    return {
        "original_time": original_time,
        "optimized_time_first": optimized_time_first,
        "optimized_time_second": optimized_time_second,
        "speedup_first": speedup_first,
        "speedup_second": speedup_second,
    }


def run_benchmark():
    """ベンチマークを実行"""
    logger.info("=== 遺伝子生成器 最適化ベンチマーク開始 ===")

    try:
        results = benchmark_gene_generators()
        logger.info("\n=== サマリー ===")
        logger.info(f"初回: {results['speedup_first']:.2f}倍高速化")
        logger.info(f"2回目: {results['speedup_second']:.2f}倍高速化")
        return results
    except Exception as e:
        logger.error(f"ベンチマークエラー: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_benchmark()
