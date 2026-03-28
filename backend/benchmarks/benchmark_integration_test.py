"""
統合テスト

最適化後の統合テストを実行します。
"""

import logging
import os
import sys
import time
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

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


def create_mock_ohlcv_data(n_bars: int = 1000) -> pd.DataFrame:
    """モックOHLCVデータを生成"""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="1h")
    close = 50000 + np.cumsum(np.random.randn(n_bars) * 100)
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(n_bars) * 50,
            "High": close + abs(np.random.randn(n_bars) * 100),
            "Low": close - abs(np.random.randn(n_bars) * 100),
            "Close": close,
            "Volume": np.random.randint(100, 10000, n_bars),
        },
        index=dates,
    )


def test_individual_evaluator():
    """IndividualEvaluatorのテスト"""
    logger.info("=== IndividualEvaluatorテスト ===")

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

    # モックバックテストサービス
    mock_backtest_service = MagicMock()
    mock_backtest_service.run_backtest.return_value = create_mock_backtest_result()

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

    # テスト遺伝子
    gene = StrategyGene(
        id="test_gene",
        indicators=[IndicatorGene(type="SMA", parameters={"period": 20})],
        long_entry_conditions=[
            Condition(left_operand="sma_20", operator=">", right_operand="close")
        ],
        short_entry_conditions=[],
        tpsl_gene=TPSLGene(method=TPSLMethod.VOLATILITY_BASED, atr_multiplier_sl=2.0),
    )

    # 評価テスト
    start_time = time.perf_counter()
    result = evaluator.evaluate_individual(gene, config)
    elapsed = time.perf_counter() - start_time

    logger.info(f"評価結果: {result}")
    logger.info(f"実行時間: {elapsed*1000:.3f}ms")

    # キャッシュテスト
    start_time = time.perf_counter()
    result2 = evaluator.evaluate_individual(gene, config)
    elapsed2 = time.perf_counter() - start_time

    logger.info(f"キャッシュ評価結果: {result2}")
    logger.info(f"キャッシュ実行時間: {elapsed2*1000:.3f}ms")

    return result == result2


def test_gene_generator():
    """RandomGeneGeneratorのテスト"""
    logger.info("\n=== RandomGeneGeneratorテスト ===")

    from app.services.auto_strategy.config.ga import GAConfig
    from app.services.auto_strategy.generators.random_gene_generator import (
        RandomGeneGenerator,
    )

    config = GAConfig()
    generator = RandomGeneGenerator(config)

    # 遺伝子生成テスト
    start_time = time.perf_counter()
    gene1 = generator.generate_random_gene()
    elapsed1 = time.perf_counter() - start_time

    logger.info(f"遺伝子生成1: {len(gene1.indicators)}個のインジケーター")
    logger.info(f"実行時間: {elapsed1*1000:.3f}ms")

    # 2回目の生成テスト
    start_time = time.perf_counter()
    gene2 = generator.generate_random_gene()
    elapsed2 = time.perf_counter() - start_time

    logger.info(f"遺伝子生成2: {len(gene2.indicators)}個のインジケーター")
    logger.info(f"実行時間: {elapsed2*1000:.3f}ms")

    return True


def test_fitness_calculator():
    """FitnessCalculatorのテスト"""
    logger.info("\n=== FitnessCalculatorテスト ===")

    from app.services.auto_strategy.config.ga import GAConfig
    from app.services.auto_strategy.core.fitness.optimized_fitness_calculator import (
        OptimizedFitnessCalculator,
    )

    calculator = OptimizedFitnessCalculator()
    config = GAConfig()

    backtest_result = create_mock_backtest_result()

    # フィットネス計算テスト
    start_time = time.perf_counter()
    fitness = calculator.calculate_fitness(backtest_result, config)
    elapsed = time.perf_counter() - start_time

    logger.info(f"フィットネス: {fitness}")
    logger.info(f"実行時間: {elapsed*1000:.3f}ms")

    return fitness > 0


def run_all_tests():
    """全テストを実行"""
    logger.info("=== 統合テスト開始 ===")

    results = {}

    try:
        results["individual_evaluator"] = test_individual_evaluator()
    except Exception as e:
        logger.error(f"IndividualEvaluatorテストエラー: {e}")
        import traceback
        traceback.print_exc()
        results["individual_evaluator"] = False

    try:
        results["gene_generator"] = test_gene_generator()
    except Exception as e:
        logger.error(f"RandomGeneGeneratorテストエラー: {e}")
        import traceback
        traceback.print_exc()
        results["gene_generator"] = False

    try:
        results["fitness_calculator"] = test_fitness_calculator()
    except Exception as e:
        logger.error(f"FitnessCalculatorテストエラー: {e}")
        import traceback
        traceback.print_exc()
        results["fitness_calculator"] = False

    logger.info("\n=== テスト結果 ===")
    for name, result in results.items():
        logger.info(f"{name}: {'成功' if result else '失敗'}")

    all_passed = all(results.values())
    logger.info(f"\n全テスト: {'成功' if all_passed else '失敗'}")

    return all_passed


if __name__ == "__main__":
    run_all_tests()
