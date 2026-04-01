"""
評価ストラテジーとデータプロバイダー 最適化前後比較ベンチマーク
"""

import logging
import os
import sys
import time
import threading
from unittest.mock import MagicMock

import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_backtest_result():
    """モックバックテスト結果を生成"""
    return {
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


def create_mock_config():
    """モックGA設定を生成"""
    from app.services.auto_strategy.config.ga import GAConfig

    config = GAConfig()
    config.enable_walk_forward = True
    config.wfa_n_folds = 3
    config.wfa_train_ratio = 0.7
    config.oos_split_ratio = 0.3
    config.oos_fitness_weight = 0.5
    return config


def benchmark_evaluation_strategy():
    """評価ストラテジーのベンチマーク"""
    logger.info("=== 評価ストラテジー 最適化前後比較 ===")

    from app.services.auto_strategy.core.evaluation.evaluation_strategies import (
        EvaluationStrategy,
    )
    OptimizedEvaluationStrategy = EvaluationStrategy

    # モックエvaluator
    mock_evaluator = MagicMock()
    mock_evaluator._perform_single_evaluation.return_value = (0.15, 1.5, 0.1)

    # オリジナル版
    original_strategy = EvaluationStrategy(mock_evaluator)

    # 最適化版
    optimized_strategy = OptimizedEvaluationStrategy(mock_evaluator, max_workers=2)

    # テストデータ
    from app.services.auto_strategy.genes import (
        IndicatorGene,
        StrategyGene,
        TPSLGene,
        Condition,
    )

    gene = StrategyGene(
        id="test_gene",
        indicators=[IndicatorGene(type="SMA", parameters={"period": 20})],
        long_entry_conditions=[
            Condition(left_operand="sma_20", operator=">", right_operand="close")
        ],
        short_entry_conditions=[],
        tpsl_gene=TPSLGene(),
    )

    config = create_mock_config()
    backtest_config = {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-04-01",
    }

    # ウームアップ
    for _ in range(3):
        original_strategy.execute(gene, backtest_config, config)
        optimized_strategy.execute(gene, backtest_config, config)

    # ベンチマーク実行
    n_iterations = 20

    # オリジナル版
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        original_strategy.execute(gene, backtest_config, config)
    original_time = time.perf_counter() - start_time

    # 最適化版
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        optimized_strategy.execute(gene, backtest_config, config)
    optimized_time = time.perf_counter() - start_time

    # 結果表示
    logger.info(f"\n=== 結果 ({n_iterations}回実行) ===")
    logger.info(f"オリジナル版: {original_time:.4f}秒 ({original_time/n_iterations*1000:.3f}ms/回)")
    logger.info(f"最適化版:    {optimized_time:.4f}秒 ({optimized_time/n_iterations*1000:.3f}ms/回)")

    speedup = original_time / optimized_time
    logger.info(f"高速化率: {speedup:.2f}倍 ({(1 - optimized_time/original_time) * 100:.1f}%削減)")

    return {
        "original_time": original_time,
        "optimized_time": optimized_time,
        "speedup": speedup,
    }


def benchmark_data_provider():
    """データプロバイダーのベンチマーク"""
    logger.info("\n=== データプロバイダー 最適化前後比較 ===")

    from cachetools import LRUCache

    from app.services.auto_strategy.core.evaluation.backtest_data_provider import (
        BacktestDataProvider,
    )
    OptimizedBacktestDataProvider = BacktestDataProvider

    # モックバックテストサービス
    mock_service = MagicMock()
    mock_service.ensure_data_service_initialized.return_value = None

    # モックデータ
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="1h")
    close = 50000 + np.cumsum(np.random.randn(1000) * 100)
    mock_data = pd.DataFrame(
        {
            "Open": close + np.random.randn(1000) * 50,
            "High": close + abs(np.random.randn(1000) * 100),
            "Low": close - abs(np.random.randn(1000) * 100),
            "Close": close,
            "Volume": np.random.randint(100, 10000, 1000),
        },
        index=dates,
    )

    mock_service.data_service.get_data_for_backtest.return_value = mock_data
    mock_service.data_service.get_ohlcv_data.return_value = mock_data

    # キャッシュとロック
    original_cache = LRUCache(maxsize=100)
    original_lock = threading.Lock()

    optimized_cache = LRUCache(maxsize=100)
    optimized_lock = threading.Lock()

    # オリジナル版
    original_provider = BacktestDataProvider(
        backtest_service=mock_service,
        data_cache=original_cache,
        lock=original_lock,
    )

    # 最適化版
    optimized_provider = OptimizedBacktestDataProvider(
        backtest_service=mock_service,
        data_cache=optimized_cache,
        lock=optimized_lock,
        prefetch_enabled=True,
        max_prefetch_workers=2,
    )

    # テストデータ
    backtest_config = {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-04-01",
    }

    # ウームアップ
    for _ in range(5):
        original_provider.get_cached_backtest_data(backtest_config)
        optimized_provider.get_cached_backtest_data(backtest_config)

    # ベンチマーク実行
    n_iterations = 100

    # オリジナル版（初回）
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        original_provider.get_cached_backtest_data(backtest_config)
    original_time_first = time.perf_counter() - start_time

    # 最適化版（初回）
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        optimized_provider.get_cached_backtest_data(backtest_config)
    optimized_time_first = time.perf_counter() - start_time

    # 最適化版（2回目 - キャッシュヒット）
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        optimized_provider.get_cached_backtest_data(backtest_config)
    optimized_time_second = time.perf_counter() - start_time

    # 結果表示
    logger.info(f"\n=== 結果 ({n_iterations}回実行) ===")
    logger.info(f"オリジナル版: {original_time_first:.4f}秒 ({original_time_first/n_iterations*1000:.3f}ms/回)")
    logger.info(f"最適化版（初回）: {optimized_time_first:.4f}秒 ({optimized_time_first/n_iterations*1000:.3f}ms/回)")
    logger.info(f"最適化版（2回目）: {optimized_time_second:.4f}秒 ({optimized_time_second/n_iterations*1000:.3f}ms/回)")

    speedup_first = original_time_first / optimized_time_first
    speedup_second = original_time_first / optimized_time_second

    logger.info(f"\n=== 高速化率 ===")
    logger.info(f"最適化版（初回）: {speedup_first:.2f}倍")
    logger.info(f"最適化版（2回目）: {speedup_second:.2f}倍")

    return {
        "original_time_first": original_time_first,
        "optimized_time_first": optimized_time_first,
        "optimized_time_second": optimized_time_second,
        "speedup_first": speedup_first,
        "speedup_second": speedup_second,
    }


def run_all_benchmarks():
    """全ベンチマークを実行"""
    logger.info("=== 評価ストラテジーとデータプロバイダー 最適化ベンチマーク開始 ===")

    all_results = {}

    try:
        all_results["evaluation_strategy"] = benchmark_evaluation_strategy()
    except Exception as e:
        logger.error(f"評価ストラテジーベンチマークエラー: {e}")
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
    if "evaluation_strategy" in all_results:
        logger.info(f"評価ストラテジー: {all_results['evaluation_strategy']['speedup']:.2f}倍高速化")
    if "data_provider" in all_results:
        logger.info(f"データプロバイダー（初回）: {all_results['data_provider']['speedup_first']:.2f}倍高速化")
        logger.info(f"データプロバイダー（2回目）: {all_results['data_provider']['speedup_second']:.2f}倍高速化")

    return all_results


if __name__ == "__main__":
    run_all_benchmarks()
