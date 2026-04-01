"""
Auto Strategy コンポーネントの結合テスト

標準経路に統合されたコンポーネントが正しく連携して動作するかを検証します。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import time
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd

from app.services.auto_strategy.config.ga import GAConfig
from app.services.auto_strategy.core.evaluation.individual_evaluator import IndividualEvaluator
from app.services.auto_strategy.core.evaluation.condition_evaluator import ConditionEvaluator
from app.services.auto_strategy.core.fitness.fitness_calculator import FitnessCalculator
from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.services.auto_strategy.genes import (
    Condition,
    IndicatorGene,
    StrategyGene,
    TPSLGene,
)
from app.services.auto_strategy.config.constants import TPSLMethod


# =============================================================================
# フィクスチャ
# =============================================================================

@pytest.fixture
def ga_config():
    """GA設定のフィクスチャ"""
    config = GAConfig()
    config.fitness_weights = {
        "total_return": 0.3,
        "sharpe_ratio": 0.4,
        "max_drawdown": 0.2,
        "win_rate": 0.1,
        "balance_score": 0.1,
    }
    config.max_indicators = 5
    config.min_indicators = 1
    config.max_conditions = 3
    config.min_conditions = 1
    return config


@pytest.fixture
def mock_backtest_result():
    """モックバックテスト結果のフィクスチャ"""
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
        "trade_history": [
            {"size": 1, "pnl": 100} if i % 2 == 0 else {"size": -1, "pnl": 50}
            for i in range(100)
        ],
        "start_date": "2024-01-01",
        "end_date": "2024-04-01",
    }


@pytest.fixture
def mock_ohlcv_data():
    """モックOHLCVデータのフィクスチャ"""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="1h")
    close = 50000 + np.cumsum(np.random.randn(1000) * 100)
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(1000) * 50,
            "High": close + abs(np.random.randn(1000) * 100),
            "Low": close - abs(np.random.randn(1000) * 100),
            "Close": close,
            "Volume": np.random.randint(100, 10000, 1000),
        },
        index=dates,
    )


@pytest.fixture
def mock_strategy(mock_ohlcv_data):
    """モック戦略インスタンスのフィクスチャ"""
    strategy = MagicMock()
    strategy.data = mock_ohlcv_data
    strategy.indicators = {
        "sma_20": mock_ohlcv_data["Close"].rolling(20).mean(),
        "ema_50": mock_ohlcv_data["Close"].ewm(span=50).mean(),
        "rsi_14": calculate_rsi(mock_ohlcv_data["Close"], 14),
    }
    return strategy


def calculate_rsi(prices, period):
    """RSIを計算"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# =============================================================================
# 結合テスト
# =============================================================================

class TestAutoStrategyComponentsIntegration:
    """Auto Strategy コンポーネントの結合テスト"""

    def test_individual_evaluator_with_fitness_calculator(
        self, ga_config, mock_backtest_result
    ):
        """IndividualEvaluator と FitnessCalculator の結合テスト"""
        mock_backtest_service = MagicMock()
        mock_backtest_service.run_backtest.return_value = mock_backtest_result

        evaluator = IndividualEvaluator(mock_backtest_service, max_cache_size=100)
        evaluator.set_backtest_config({
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-04-01",
        })

        gene = StrategyGene(
            id="test_gene",
            indicators=[IndicatorGene(type="SMA", parameters={"period": 20})],
            long_entry_conditions=[
                Condition(left_operand="sma_20", operator=">", right_operand="close")
            ],
            short_entry_conditions=[],
            tpsl_gene=TPSLGene(method=TPSLMethod.VOLATILITY_BASED, atr_multiplier_sl=2.0),
        )

        # 評価実行
        start_time = time.perf_counter()
        result = evaluator.evaluate_individual(gene, ga_config)
        elapsed = time.perf_counter() - start_time

        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) > 0
        assert elapsed < 1.0  # 1秒以内に完了

    def test_condition_evaluator_with_fitness_calculator(
        self, ga_config, mock_strategy
    ):
        """ConditionEvaluator と FitnessCalculator の結合テスト"""
        condition_evaluator = ConditionEvaluator()
        fitness_calculator = FitnessCalculator()

        # 条件評価
        condition = Condition(
            left_operand="sma_20", operator=">", right_operand="ema_50"
        )

        start_time = time.perf_counter()
        condition_result = condition_evaluator.evaluate_single_condition(
            condition, mock_strategy
        )
        condition_elapsed = time.perf_counter() - start_time

        assert isinstance(condition_result, bool)
        assert condition_elapsed < 0.1  # 100ms以内に完了

    def test_gene_generation_with_fitness_evaluation(self, ga_config):
        """遺伝子生成とフィットネス評価の結合テスト"""
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
        evaluator.set_backtest_config({
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-04-01",
        })

        generator = RandomGeneGenerator(ga_config)

        # 遺伝子生成
        start_time = time.perf_counter()
        gene = generator.generate_random_gene()
        generation_elapsed = time.perf_counter() - start_time

        assert gene is not None
        assert isinstance(gene, StrategyGene)
        assert generation_elapsed < 5.0  # 5秒以内に完了

        # フィットネス評価
        start_time = time.perf_counter()
        fitness = evaluator.evaluate_individual(gene, ga_config)
        evaluation_elapsed = time.perf_counter() - start_time

        assert fitness is not None
        assert isinstance(fitness, tuple)
        assert evaluation_elapsed < 1.0  # 1秒以内に完了

    def test_multiple_gene_evaluation(self, ga_config):
        """複数遺伝子評価の結合テスト"""
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
        evaluator.set_backtest_config({
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-04-01",
        })

        generator = RandomGeneGenerator(ga_config)

        # 複数遺伝子を生成・評価
        genes = []
        for i in range(10):
            gene = generator.generate_random_gene()
            gene.id = f"gene_{i}"
            genes.append(gene)

        # 評価実行
        start_time = time.perf_counter()
        results = []
        for gene in genes:
            fitness = evaluator.evaluate_individual(gene, ga_config)
            results.append(fitness)
        total_elapsed = time.perf_counter() - start_time

        assert len(results) == 10
        assert all(isinstance(r, tuple) for r in results)
        assert total_elapsed < 10.0  # 10秒以内に完了

    def test_cache_effectiveness(self, ga_config):
        """キャッシュ効果のテスト"""
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
        evaluator.set_backtest_config({
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-04-01",
        })

        gene = StrategyGene(
            id="cache_test_gene",
            indicators=[IndicatorGene(type="SMA", parameters={"period": 20})],
            long_entry_conditions=[
                Condition(left_operand="sma_20", operator=">", right_operand="close")
            ],
            short_entry_conditions=[],
            tpsl_gene=TPSLGene(method=TPSLMethod.VOLATILITY_BASED, atr_multiplier_sl=2.0),
        )

        # 初回評価
        start_time = time.perf_counter()
        result1 = evaluator.evaluate_individual(gene, ga_config)
        first_elapsed = time.perf_counter() - start_time

        # 2回目評価（キャッシュヒット）
        start_time = time.perf_counter()
        result2 = evaluator.evaluate_individual(gene, ga_config)
        second_elapsed = time.perf_counter() - start_time

        assert result1 == result2
        assert second_elapsed < first_elapsed * 0.1  # 2回目は10倍以上高速


class TestPerformanceBenchmark:
    """パフォーマンスベンチマークテスト"""

    def test_fitness_calculation_performance(self, ga_config, mock_backtest_result):
        """フィットネス計算のパフォーマンステスト"""
        calculator = FitnessCalculator()

        # ウームアップ
        for _ in range(10):
            calculator.calculate_fitness(mock_backtest_result, ga_config)

        # ベンチマーク
        iterations = 1000
        start_time = time.perf_counter()
        for _ in range(iterations):
            calculator.calculate_fitness(mock_backtest_result, ga_config)
        elapsed = time.perf_counter() - start_time

        ms_per_iteration = elapsed / iterations * 1000
        assert ms_per_iteration < 1.0  # 1ms以内

    def test_condition_evaluation_performance(self, mock_strategy):
        """条件評価のパフォーマンステスト"""
        evaluator = ConditionEvaluator()
        condition = Condition(
            left_operand="sma_20", operator=">", right_operand="ema_50"
        )

        # ウームアップ
        for _ in range(100):
            evaluator.evaluate_single_condition(condition, mock_strategy)

        # ベンチマーク
        iterations = 1000
        start_time = time.perf_counter()
        for _ in range(iterations):
            evaluator.evaluate_single_condition(condition, mock_strategy)
        elapsed = time.perf_counter() - start_time

        ms_per_iteration = elapsed / iterations * 1000
        assert ms_per_iteration < 0.1  # 0.1ms以内

    def test_gene_generation_performance(self, ga_config):
        """遺伝子生成のパフォーマンステスト"""
        generator = RandomGeneGenerator(ga_config)

        # ウームアップ
        for _ in range(5):
            generator.generate_random_gene()

        # ベンチマーク
        iterations = 100
        start_time = time.perf_counter()
        for _ in range(iterations):
            generator.generate_random_gene()
        elapsed = time.perf_counter() - start_time

        ms_per_iteration = elapsed / iterations * 1000
        assert ms_per_iteration < 100.0  # 100ms以内
