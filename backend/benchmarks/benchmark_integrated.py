"""
統合後ベンチマーク

最適化後の統合ベンチマークを実行します。
"""

import logging
import os
import sys
import time
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_backtest_result(n_trades: int = 100) -> Dict[str, Any]:
    """モックバックテスト結果を生成"""
    np.random.seed(42)

    trade_history = []
    for _ in range(n_trades):
        size = np.random.choice([1, -1])
        pnl = np.random.random() * 200 - 50
        trade_history.append({"size": size, "pnl": pnl})

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


def benchmark_integrated_fitness():
    """統合後のフィットネス計算ベンチマーク"""
    logger.info("=== 統合後フィットネス計算ベンチマーク ===")

    from app.services.auto_strategy.config.ga import GAConfig
    from app.services.auto_strategy.core.evaluation.individual_evaluator import (
        IndividualEvaluator,
    )

    # モックバックテストサービス
    mock_backtest_service = MagicMock()
    mock_backtest_service.run_backtest.return_value = create_mock_backtest_result(100)

    evaluator = IndividualEvaluator(mock_backtest_service, max_cache_size=100)
    config = GAConfig()

    # バックテスト設定
    backtest_config = {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-04-01",
    }
    evaluator.set_backtest_config(backtest_config)

    # モック遺伝子
    from app.services.auto_strategy.genes import (
        Condition,
        IndicatorGene,
        StrategyGene,
        TPSLGene,
    )
    from app.services.auto_strategy.config.constants import TPSLMethod

    gene = StrategyGene(
        id="test_gene",
        indicators=[
            IndicatorGene(type="SMA", parameters={"period": 20}),
            IndicatorGene(type="EMA", parameters={"period": 50}),
        ],
        long_entry_conditions=[
            Condition(left_operand="sma_20", operator=">", right_operand="ema_50")
        ],
        short_entry_conditions=[
            Condition(left_operand="sma_20", operator="<", right_operand="ema_50")
        ],
            tpsl_gene=TPSLGene(method=TPSLMethod.VOLATILITY_BASED, atr_multiplier_sl=2.0, atr_multiplier_tp=3.0),
    )

    # ウームアップ
    for _ in range(5):
        evaluator.evaluate_individual(gene, config)

    # ベンチマーク実行
    n_iterations = 100

    start_time = time.perf_counter()
    for _ in range(n_iterations):
        evaluator.evaluate_individual(gene, config)
    elapsed = time.perf_counter() - start_time

    logger.info(f"\n=== 結果 ({n_iterations}回実行) ===")
    logger.info(f"実行時間: {elapsed:.4f}秒 ({elapsed/n_iterations*1000:.3f}ms/回)")

    # キャッシュ統計
    cache_info = evaluator.get_cache_info()
    logger.info(f"\n=== キャッシュ統計 ===")
    logger.info(f"ヒット数: {cache_info['cache_hits']}")
    logger.info(f"ミス数: {cache_info['cache_misses']}")
    total = cache_info['cache_hits'] + cache_info['cache_misses']
    hit_rate = cache_info['cache_hits'] / total if total > 0 else 0
    logger.info(f"ヒット率: {hit_rate*100:.1f}%")

    return {
        "elapsed": elapsed,
        "ms_per_iteration": elapsed / n_iterations * 1000,
        "cache_hit_rate": hit_rate,
    }


def benchmark_with_different_trade_counts():
    """異なるトレード数でのベンチマーク"""
    logger.info("\n=== トレード数別ベンチマーク ===")

    from app.services.auto_strategy.config.ga import GAConfig
    from app.services.auto_strategy.core.evaluation.individual_evaluator import (
        IndividualEvaluator,
    )
    from app.services.auto_strategy.genes import (
        Condition,
        IndicatorGene,
        StrategyGene,
        TPSLGene,
    )
    from app.services.auto_strategy.config.constants import TPSLMethod

    trade_counts = [10, 50, 100, 500]
    n_iterations = 50

    results = {}

    for n_trades in trade_counts:
        # モックバックテストサービス
        mock_backtest_service = MagicMock()
        mock_backtest_service.run_backtest.return_value = create_mock_backtest_result(n_trades)

        evaluator = IndividualEvaluator(mock_backtest_service, max_cache_size=100)
        config = GAConfig()

        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-04-01",
        }
        evaluator.set_backtest_config(backtest_config)

        gene = StrategyGene(
            id=f"test_gene_{n_trades}",
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}),
            ],
            long_entry_conditions=[
                Condition(left_operand="sma_20", operator=">", right_operand="close")
            ],
            short_entry_conditions=[
                Condition(left_operand="sma_20", operator="<", right_operand="close")
            ],
        tpsl_gene=TPSLGene(method=TPSLMethod.VOLATILITY_BASED, atr_multiplier_sl=2.0, atr_multiplier_tp=3.0),
        )

        # ウームアップ
        for _ in range(3):
            evaluator.evaluate_individual(gene, config)

        # ベンチマーク
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            evaluator.evaluate_individual(gene, config)
        elapsed = time.perf_counter() - start_time

        results[n_trades] = {
            "elapsed": elapsed,
            "ms_per_iteration": elapsed / n_iterations * 1000,
        }

        logger.info(f"\nトレード数 {n_trades}:")
        logger.info(f"  実行時間: {elapsed:.4f}秒 ({elapsed/n_iterations*1000:.3f}ms/回)")

    return results


def run_all_benchmarks():
    """全ベンチマークを実行"""
    logger.info("=== 統合後ベンチマーク開始 ===")

    all_results = {}

    try:
        all_results["integrated_fitness"] = benchmark_integrated_fitness()
    except Exception as e:
        logger.error(f"統合フィットネスベンチマークエラー: {e}")
        import traceback
        traceback.print_exc()

    try:
        all_results["trade_counts"] = benchmark_with_different_trade_counts()
    except Exception as e:
        logger.error(f"トレード数別ベンチマークエラー: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\n=== 全ベンチマーク完了 ===")

    # サマリー
    if "integrated_fitness" in all_results:
        result = all_results["integrated_fitness"]
        logger.info(f"\n=== サマリー ===")
        logger.info(f"フィットネス計算: {result['ms_per_iteration']:.3f}ms/回")
        logger.info(f"キャッシュヒット率: {result['cache_hit_rate']*100:.1f}%")

    return all_results


if __name__ == "__main__":
    run_all_benchmarks()
