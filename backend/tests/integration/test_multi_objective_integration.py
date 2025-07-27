"""
多目的最適化GA統合テスト

実際のGAエンジンを使用した多目的最適化の動作確認を行います。
"""

import pytest
import logging
from unittest.mock import Mock, patch
from typing import Dict, Any

from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.engines.ga_engine import GeneticAlgorithmEngine
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.core.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)

logger = logging.getLogger(__name__)


class TestMultiObjectiveIntegration:
    """多目的最適化GA統合テストクラス"""

    @pytest.fixture
    def mock_backtest_service(self):
        """モックバックテストサービス"""
        mock_service = Mock()

        # 多様なバックテスト結果を返すモック
        def mock_run_backtest(config):
            import random

            return {
                "performance_metrics": {
                    "total_return": random.uniform(0.05, 0.25),  # 5-25%
                    "sharpe_ratio": random.uniform(0.5, 2.5),  # 0.5-2.5
                    "max_drawdown": random.uniform(0.02, 0.15),  # 2-15%
                    "win_rate": random.uniform(0.45, 0.75),  # 45-75%
                    "profit_factor": random.uniform(1.1, 2.5),  # 1.1-2.5
                    "total_trades": random.randint(20, 100),  # 20-100取引
                    "long_trades": random.randint(10, 50),
                    "short_trades": random.randint(10, 50),
                    "long_pnl": random.uniform(0.05, 0.15),
                    "short_pnl": random.uniform(0.05, 0.15),
                }
            }

        mock_service.run_backtest.side_effect = mock_run_backtest
        return mock_service

    @pytest.fixture
    def strategy_factory(self):
        """戦略ファクトリー"""
        return StrategyFactory()

    @pytest.fixture
    def gene_generator(self):
        """遺伝子生成器"""
        config = GAConfig()
        return RandomGeneGenerator(config)

    def test_single_objective_compatibility(
        self, mock_backtest_service, strategy_factory, gene_generator
    ):
        """単一目的最適化との互換性テスト"""
        # 単一目的設定
        config = GAConfig(
            population_size=5, generations=2, enable_multi_objective=False
        )

        # GAエンジン初期化
        ga_engine = GeneticAlgorithmEngine(
            mock_backtest_service, strategy_factory, gene_generator
        )

        # バックテスト設定
        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-01-31",
            "initial_capital": 10000,
            "commission_rate": 0.001,
        }

        # 進化実行
        result = ga_engine.run_evolution(config, backtest_config)

        # 結果確認
        assert "best_strategy" in result
        assert "best_fitness" in result
        assert isinstance(result["best_fitness"], float)  # 単一値
        assert "pareto_front" not in result  # 多目的結果は含まれない
        assert result["generations_completed"] == config.generations

    def test_multi_objective_basic(
        self, mock_backtest_service, strategy_factory, gene_generator
    ):
        """基本的な多目的最適化テスト"""
        # 多目的設定
        config = GAConfig.create_multi_objective(
            objectives=["total_return", "max_drawdown"], weights=[1.0, -1.0]
        )
        config.population_size = 8
        config.generations = 3

        # GAエンジン初期化
        ga_engine = GeneticAlgorithmEngine(
            mock_backtest_service, strategy_factory, gene_generator
        )

        # バックテスト設定
        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-01-31",
            "initial_capital": 10000,
            "commission_rate": 0.001,
        }

        # 進化実行
        result = ga_engine.run_evolution(config, backtest_config)

        # 結果確認
        assert "best_strategy" in result
        assert "best_fitness" in result
        assert isinstance(result["best_fitness"], (list, tuple))  # 複数値
        assert len(result["best_fitness"]) == 2  # 2つの目的
        assert "pareto_front" in result  # パレート最適解が含まれる
        assert "objectives" in result
        assert result["objectives"] == ["total_return", "max_drawdown"]
        assert isinstance(result["pareto_front"], list)
        assert len(result["pareto_front"]) > 0  # パレート最適解が存在

    def test_multi_objective_three_objectives(
        self, mock_backtest_service, strategy_factory, gene_generator
    ):
        """3目的最適化テスト"""
        # 3目的設定
        config = GAConfig.create_multi_objective(
            objectives=["total_return", "sharpe_ratio", "max_drawdown"],
            weights=[1.0, 1.0, -1.0],
        )
        config.population_size = 6
        config.generations = 2

        # GAエンジン初期化
        ga_engine = GeneticAlgorithmEngine(
            mock_backtest_service, strategy_factory, gene_generator
        )

        # バックテスト設定
        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "4h",
            "start_date": "2023-01-01",
            "end_date": "2023-01-31",
            "initial_capital": 10000,
            "commission_rate": 0.001,
        }

        # 進化実行
        result = ga_engine.run_evolution(config, backtest_config)

        # 結果確認
        assert len(result["best_fitness"]) == 3  # 3つの目的
        assert result["objectives"] == ["total_return", "sharpe_ratio", "max_drawdown"]

        # パレート最適解の確認
        pareto_front = result["pareto_front"]
        assert len(pareto_front) > 0

        for solution in pareto_front:
            assert "strategy" in solution
            assert "fitness_values" in solution
            assert len(solution["fitness_values"]) == 3

    def test_pareto_front_quality(
        self, mock_backtest_service, strategy_factory, gene_generator
    ):
        """パレート最適解の品質テスト"""
        # 多目的設定
        config = GAConfig.create_multi_objective()
        config.population_size = 10
        config.generations = 3

        # GAエンジン初期化
        ga_engine = GeneticAlgorithmEngine(
            mock_backtest_service, strategy_factory, gene_generator
        )

        # バックテスト設定
        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-01-31",
            "initial_capital": 10000,
            "commission_rate": 0.001,
        }

        # 進化実行
        result = ga_engine.run_evolution(config, backtest_config)

        # パレート最適解の品質確認
        pareto_front = result["pareto_front"]
        assert len(pareto_front) <= 10  # 最大10個

        # 各解が有効な値を持つことを確認
        for solution in pareto_front:
            fitness_values = solution["fitness_values"]
            assert len(fitness_values) == 2
            assert all(isinstance(val, (int, float)) for val in fitness_values)
            assert fitness_values[0] >= 0  # total_return >= 0
            assert fitness_values[1] >= 0  # max_drawdown >= 0

    def test_error_handling(self, strategy_factory, gene_generator):
        """エラーハンドリングテスト"""
        # エラーを発生させるモックバックテストサービス
        mock_service = Mock()
        mock_service.run_backtest.side_effect = Exception("Backtest error")

        # 多目的設定
        config = GAConfig.create_multi_objective()
        config.population_size = 3
        config.generations = 1

        # GAエンジン初期化
        ga_engine = GeneticAlgorithmEngine(
            mock_service, strategy_factory, gene_generator
        )

        # バックテスト設定
        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-01-31",
            "initial_capital": 10000,
            "commission_rate": 0.001,
        }

        # 進化実行（エラーが発生しても処理が継続されることを確認）
        result = ga_engine.run_evolution(config, backtest_config)

        # 結果確認（エラー時のデフォルト値が設定される）
        assert "best_strategy" in result
        assert "best_fitness" in result


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO)

    # テスト実行
    pytest.main([__file__, "-v", "-s"])
