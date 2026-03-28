"""
その他コンポーネントの最適化ベンチマーク

遺伝子生成、条件評価、交叉・突然変異のパフォーマンスを測定します。
"""

import logging
import os
import sys
import time
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

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


def benchmark_gene_generation():
    """遺伝子生成のベンチマーク"""
    logger.info("=== 遺伝子生成ベンチマーク ===")

    from app.services.auto_strategy.generators.random_gene_generator import (
        RandomGeneGenerator,
    )

    config = create_mock_config()
    generator = RandomGeneGenerator(config)

    # ウームアップ
    for _ in range(5):
        generator.generate_random_gene()

    # ベンチマーク実行
    n_iterations = 100

    start_time = time.perf_counter()
    for _ in range(n_iterations):
        generator.generate_random_gene()
    elapsed = time.perf_counter() - start_time

    logger.info(f"\n=== 結果 ({n_iterations}回実行) ===")
    logger.info(f"実行時間: {elapsed:.4f}秒 ({elapsed/n_iterations*1000:.3f}ms/回)")

    return {
        "elapsed": elapsed,
        "ms_per_iteration": elapsed / n_iterations * 1000,
    }


def benchmark_condition_evaluation():
    """条件評価のベンチマーク"""
    logger.info("\n=== 条件評価ベンチマーク ===")

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

    # 単一条件評価
    condition = Condition(
        left_operand="sma_20", operator=">", right_operand="ema_50"
    )

    # ウームアップ
    for _ in range(100):
        evaluator.evaluate_single_condition(condition, strategy)

    # ベンチマーク実行
    n_iterations = 1000

    start_time = time.perf_counter()
    for _ in range(n_iterations):
        evaluator.evaluate_single_condition(condition, strategy)
    elapsed = time.perf_counter() - start_time

    logger.info(f"\n=== 結果 ({n_iterations}回実行) ===")
    logger.info(f"実行時間: {elapsed:.4f}秒 ({elapsed/n_iterations*1000:.3f}ms/回)")

    # ベクトル化評価
    start_time = time.perf_counter()
    for _ in range(100):
        evaluator.evaluate_single_condition_vectorized(condition, strategy)
    elapsed_vectorized = time.perf_counter() - start_time

    logger.info(f"\n=== ベクトル化評価 (100回実行) ===")
    logger.info(f"実行時間: {elapsed_vectorized:.4f}秒 ({elapsed_vectorized/100*1000:.3f}ms/回)")

    return {
        "elapsed": elapsed,
        "ms_per_iteration": elapsed / n_iterations * 1000,
        "elapsed_vectorized": elapsed_vectorized,
        "ms_per_iteration_vectorized": elapsed_vectorized / 100 * 1000,
    }


def benchmark_crossover_mutation():
    """交叉・突然変異のベンチマーク"""
    logger.info("\n=== 交叉・突然変異ベンチマーク ===")

    from app.services.auto_strategy.config.ga import GAConfig
    from app.services.auto_strategy.genes import StrategyGene
    from app.services.auto_strategy.genes.strategy_operators import (
        crossover_strategy_genes,
        mutate_strategy_gene,
    )

    config = GAConfig()

    # テスト用遺伝子を生成
    from app.services.auto_strategy.generators.random_gene_generator import (
        RandomGeneGenerator,
    )

    generator = RandomGeneGenerator(config)
    gene1 = generator.generate_random_gene()
    gene2 = generator.generate_random_gene()

    # 交叉ベンチマーク
    n_iterations = 100

    start_time = time.perf_counter()
    for _ in range(n_iterations):
        crossover_strategy_genes(StrategyGene, gene1, gene2, config)
    elapsed_crossover = time.perf_counter() - start_time

    logger.info(f"\n=== 交叉 ({n_iterations}回実行) ===")
    logger.info(f"実行時間: {elapsed_crossover:.4f}秒 ({elapsed_crossover/n_iterations*1000:.3f}ms/回)")

    # 突然変異ベンチマーク
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        mutate_strategy_gene(gene1, config, mutation_rate=0.1)
    elapsed_mutation = time.perf_counter() - start_time

    logger.info(f"\n=== 突然変異 ({n_iterations}回実行) ===")
    logger.info(f"実行時間: {elapsed_mutation:.4f}秒 ({elapsed_mutation/n_iterations*1000:.3f}ms/回)")

    return {
        "crossover_ms": elapsed_crossover / n_iterations * 1000,
        "mutation_ms": elapsed_mutation / n_iterations * 1000,
    }


def benchmark_data_provider():
    """バックテストデータプロバイダーのベンチマーク"""
    logger.info("\n=== データプロバイダーベンチマーク ===")

    from app.services.auto_strategy.core.evaluation.backtest_data_provider import (
        BacktestDataProvider,
    )

    # モックバックテストサービス
    mock_service = MagicMock()
    mock_service.get_ohlcv_data.return_value = create_mock_ohlcv_data(1000)

    from cachetools import LRUCache
    import threading

    data_cache = LRUCache(maxsize=100)
    lock = threading.Lock()

    provider = BacktestDataProvider(
        backtest_service=mock_service,
        data_cache=data_cache,
        lock=lock,
    )

    backtest_config = {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-04-01",
    }

    # ウームアップ
    for _ in range(5):
        provider.get_cached_backtest_data(backtest_config)

    # ベンチマーク実行
    n_iterations = 100

    start_time = time.perf_counter()
    for _ in range(n_iterations):
        provider.get_cached_backtest_data(backtest_config)
    elapsed = time.perf_counter() - start_time

    logger.info(f"\n=== 結果 ({n_iterations}回実行) ===")
    logger.info(f"実行時間: {elapsed:.4f}秒 ({elapsed/n_iterations*1000:.3f}ms/回)")

    return {
        "elapsed": elapsed,
        "ms_per_iteration": elapsed / n_iterations * 1000,
    }


def run_all_benchmarks():
    """全ベンチマークを実行"""
    logger.info("=== その他コンポーネント最適化ベンチマーク開始 ===")

    all_results = {}

    try:
        all_results["gene_generation"] = benchmark_gene_generation()
    except Exception as e:
        logger.error(f"遺伝子生成ベンチマークエラー: {e}")
        import traceback
        traceback.print_exc()

    try:
        all_results["condition_evaluation"] = benchmark_condition_evaluation()
    except Exception as e:
        logger.error(f"条件評価ベンチマークエラー: {e}")
        import traceback
        traceback.print_exc()

    try:
        all_results["crossover_mutation"] = benchmark_crossover_mutation()
    except Exception as e:
        logger.error(f"交叉・突然変異ベンチマークエラー: {e}")
        import traceback
        traceback.print_exc()

    try:
        all_results["data_provider"] = benchmark_data_provider()
    except Exception as e:
        logger.error(f"データプロバイダーベンチマークエラー: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\n=== 全ベンチマーク完了 ===")

    # サマリー
    logger.info(f"\n=== サマリー ===")
    if "gene_generation" in all_results:
        logger.info(f"遺伝子生成: {all_results['gene_generation']['ms_per_iteration']:.3f}ms/回")
    if "condition_evaluation" in all_results:
        logger.info(f"条件評価: {all_results['condition_evaluation']['ms_per_iteration']:.3f}ms/回")
    if "crossover_mutation" in all_results:
        logger.info(f"交叉: {all_results['crossover_mutation']['crossover_ms']:.3f}ms/回")
        logger.info(f"突然変異: {all_results['crossover_mutation']['mutation_ms']:.3f}ms/回")
    if "data_provider" in all_results:
        logger.info(f"データプロバイダー: {all_results['data_provider']['ms_per_iteration']:.3f}ms/回")

    return all_results


if __name__ == "__main__":
    run_all_benchmarks()
