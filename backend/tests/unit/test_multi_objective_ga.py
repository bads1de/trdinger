"""
多目的最適化GA機能のテスト

NSGA-IIアルゴリズムと多目的評価機能の動作確認を行います。
"""

import pytest
import logging
from unittest.mock import Mock, patch
from typing import Dict, Any

from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.engines.deap_setup import DEAPSetup
from app.core.services.auto_strategy.engines.individual_evaluator import (
    IndividualEvaluator,
)
from app.core.services.auto_strategy.engines.ga_engine import GeneticAlgorithmEngine

logger = logging.getLogger(__name__)


class TestMultiObjectiveGA:
    """多目的最適化GAのテストクラス"""

    def test_ga_config_multi_objective(self):
        """GAConfig多目的最適化設定のテスト"""
        # デフォルト設定（単一目的）
        config = GAConfig()
        assert config.enable_multi_objective is False
        assert config.objectives == ["total_return"]
        assert config.objective_weights == [1.0]

        # 多目的最適化設定
        multi_config = GAConfig.create_multi_objective()
        assert multi_config.enable_multi_objective is True
        assert multi_config.objectives == ["total_return", "max_drawdown"]
        assert multi_config.objective_weights == [1.0, -1.0]

        # カスタム多目的設定
        custom_config = GAConfig.create_multi_objective(
            objectives=["sharpe_ratio", "win_rate", "max_drawdown"],
            weights=[1.0, 1.0, -1.0],
        )
        assert custom_config.objectives == ["sharpe_ratio", "win_rate", "max_drawdown"]
        assert custom_config.objective_weights == [1.0, 1.0, -1.0]

    def test_deap_setup_multi_objective(self):
        """DEAP設定の多目的最適化対応テスト"""
        setup = DEAPSetup()

        # 多目的最適化設定
        config = GAConfig.create_multi_objective(
            objectives=["total_return", "max_drawdown"], weights=[1.0, -1.0]
        )

        # モック関数
        mock_create_individual = Mock()
        mock_evaluate = Mock()
        mock_crossover = Mock()
        mock_mutate = Mock()

        # DEAP設定実行
        setup.setup_deap(
            config, mock_create_individual, mock_evaluate, mock_crossover, mock_mutate
        )

        # ツールボックスが正しく設定されているか確認
        toolbox = setup.get_toolbox()
        assert toolbox is not None

        # 個体クラスが正しく設定されているか確認
        Individual = setup.get_individual_class()
        assert Individual is not None

        # フィットネスクラスの重みが正しく設定されているか確認
        individual = Individual([1, 2, 3])
        assert hasattr(individual.fitness, "weights")
        assert individual.fitness.weights == (1.0, -1.0)

    def test_individual_evaluator_multi_objective(self):
        """個体評価器の多目的最適化対応テスト"""
        # モックバックテストサービス
        mock_backtest_service = Mock()
        mock_backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,  # 15%
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.08,  # 8%
                "win_rate": 0.65,  # 65%
                "profit_factor": 1.8,
                "total_trades": 50,
            }
        }
        mock_backtest_service.run_backtest.return_value = mock_backtest_result

        evaluator = IndividualEvaluator(mock_backtest_service)

        # 多目的最適化設定
        config = GAConfig.create_multi_objective(
            objectives=["total_return", "max_drawdown", "sharpe_ratio"],
            weights=[1.0, -1.0, 1.0],
        )

        # モック個体
        mock_individual = [1, 2, 3, 4, 5]

        # バックテスト設定をセット
        evaluator.set_backtest_config(
            {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "initial_capital": 10000,
            }
        )

        # 評価実行
        with patch(
            "app.core.services.auto_strategy.models.gene_encoding.GeneEncoder"
        ) as mock_encoder:
            mock_gene = Mock()
            mock_gene.id = "test_gene_123"
            mock_gene.to_dict.return_value = {"test": "data"}
            mock_encoder.return_value.decode_list_to_strategy_gene.return_value = (
                mock_gene
            )

            fitness_values = evaluator.evaluate_individual(mock_individual, config)

        # 結果確認
        assert isinstance(fitness_values, tuple)
        assert len(fitness_values) == 3  # 3つの目的
        assert fitness_values[0] == 0.15  # total_return
        assert fitness_values[1] == 0.08  # max_drawdown（符号反転なし、DEAP側で処理）
        assert fitness_values[2] == 1.5  # sharpe_ratio

    def test_multi_objective_fitness_calculation(self):
        """多目的フィットネス計算のテスト"""
        mock_backtest_service = Mock()
        evaluator = IndividualEvaluator(mock_backtest_service)

        # テストデータ
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.20,
                "sharpe_ratio": 2.0,
                "max_drawdown": 0.05,
                "win_rate": 0.70,
                "profit_factor": 2.5,
                "sortino_ratio": 2.8,
                "calmar_ratio": 4.0,
                "total_trades": 100,
            }
        }

        # 設定
        config = GAConfig.create_multi_objective(
            objectives=["total_return", "sharpe_ratio", "max_drawdown", "win_rate"],
            weights=[1.0, 1.0, -1.0, 1.0],
        )

        # フィットネス計算
        fitness_values = evaluator._calculate_multi_objective_fitness(
            backtest_result, config
        )

        # 結果確認
        assert len(fitness_values) == 4
        assert fitness_values[0] == 0.20  # total_return
        assert fitness_values[1] == 2.0  # sharpe_ratio
        assert fitness_values[2] == 0.05  # max_drawdown
        assert fitness_values[3] == 0.70  # win_rate

    def test_multi_objective_fitness_no_trades(self):
        """取引回数0の場合のフィットネス計算テスト"""
        mock_backtest_service = Mock()
        evaluator = IndividualEvaluator(mock_backtest_service)

        # 取引回数0のテストデータ
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
            }
        }

        config = GAConfig.create_multi_objective()

        # フィットネス計算
        fitness_values = evaluator._calculate_multi_objective_fitness(
            backtest_result, config
        )

        # 結果確認（取引回数0の場合は低い値が設定される）
        assert len(fitness_values) == 2
        assert all(value == 0.1 for value in fitness_values)

    def test_unknown_objective(self):
        """未知の目的に対するエラーハンドリングテスト"""
        mock_backtest_service = Mock()
        evaluator = IndividualEvaluator(mock_backtest_service)

        backtest_result = {
            "performance_metrics": {"total_return": 0.15, "total_trades": 50}
        }

        # 未知の目的を含む設定
        config = GAConfig(
            enable_multi_objective=True,
            objectives=["total_return", "unknown_metric"],
            objective_weights=[1.0, 1.0],
        )

        # フィットネス計算
        fitness_values = evaluator._calculate_multi_objective_fitness(
            backtest_result, config
        )

        # 結果確認
        assert len(fitness_values) == 2
        assert fitness_values[0] == 0.15  # total_return
        assert fitness_values[1] == 0.0  # unknown_metric（デフォルト値）


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO)

    # テスト実行
    pytest.main([__file__, "-v"])
