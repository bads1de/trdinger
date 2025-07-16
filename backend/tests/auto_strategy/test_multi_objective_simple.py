"""
多目的最適化GA 簡単なテスト

基本的な多目的最適化機能の動作確認を行います。
"""

import pytest
import logging
from unittest.mock import Mock, patch
from typing import Dict, Any

from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.engines.ga_engine import GeneticAlgorithmEngine
from app.core.services.auto_strategy.engines.deap_setup import DEAPSetup
from app.core.services.auto_strategy.engines.individual_evaluator import IndividualEvaluator
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator

logger = logging.getLogger(__name__)


class TestMultiObjectiveSimple:
    """多目的最適化GA 簡単なテストクラス"""

    def test_multi_objective_config_creation(self):
        """多目的最適化設定の作成テスト"""
        # デフォルト多目的設定
        config = GAConfig.create_multi_objective()
        
        assert config.enable_multi_objective is True
        assert config.objectives == ["total_return", "max_drawdown"]
        assert config.objective_weights == [1.0, -1.0]
        assert config.population_size == 50
        assert config.generations == 30

        # カスタム多目的設定
        custom_config = GAConfig.create_multi_objective(
            objectives=["sharpe_ratio", "win_rate", "max_drawdown"],
            weights=[1.0, 1.0, -1.0]
        )
        
        assert custom_config.objectives == ["sharpe_ratio", "win_rate", "max_drawdown"]
        assert custom_config.objective_weights == [1.0, 1.0, -1.0]

    def test_deap_setup_multi_objective_configuration(self):
        """DEAP設定の多目的最適化対応テスト"""
        setup = DEAPSetup()
        
        # 多目的設定
        config = GAConfig.create_multi_objective(
            objectives=["total_return", "sharpe_ratio"],
            weights=[1.0, 1.0]
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
        assert individual.fitness.weights == (1.0, 1.0)

    def test_multi_objective_fitness_evaluation(self):
        """多目的フィットネス評価のテスト"""
        # モックバックテストサービス
        mock_backtest_service = Mock()
        mock_backtest_result = {
            "performance_metrics": {
                "total_return": 0.20,
                "sharpe_ratio": 1.8,
                "max_drawdown": 0.06,
                "win_rate": 0.68,
                "profit_factor": 2.2,
                "total_trades": 75,
            }
        }
        mock_backtest_service.run_backtest.return_value = mock_backtest_result
        
        evaluator = IndividualEvaluator(mock_backtest_service)
        
        # 多目的設定
        config = GAConfig.create_multi_objective(
            objectives=["total_return", "sharpe_ratio", "max_drawdown"],
            weights=[1.0, 1.0, -1.0]
        )
        
        # バックテスト設定をセット
        evaluator.set_backtest_config({
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 10000,
        })
        
        # フィットネス計算のテスト
        fitness_values = evaluator._calculate_multi_objective_fitness(
            mock_backtest_result, config
        )
        
        # 結果確認
        assert isinstance(fitness_values, tuple)
        assert len(fitness_values) == 3
        assert fitness_values[0] == 0.20  # total_return
        assert fitness_values[1] == 1.8   # sharpe_ratio
        assert fitness_values[2] == 0.06  # max_drawdown

    def test_ga_engine_multi_objective_setup(self):
        """GAエンジンの多目的最適化セットアップテスト"""
        # モックサービス
        mock_backtest_service = Mock()
        mock_strategy_factory = Mock()
        mock_gene_generator = Mock()
        
        # GAエンジン初期化
        ga_engine = GeneticAlgorithmEngine(
            mock_backtest_service,
            mock_strategy_factory,
            mock_gene_generator
        )
        
        # 多目的設定
        config = GAConfig.create_multi_objective()
        config.population_size = 4  # テスト用に小さく設定
        config.generations = 1
        
        # DEAP設定のテスト
        ga_engine.setup_deap(config)
        
        # ツールボックスが正しく設定されているか確認
        toolbox = ga_engine.deap_setup.get_toolbox()
        assert toolbox is not None
        
        # 多目的最適化用の選択アルゴリズムが設定されているか確認
        assert hasattr(toolbox, 'select')

    def test_nsga2_algorithm_availability(self):
        """NSGA-IIアルゴリズムの利用可能性テスト"""
        from deap import tools
        
        # NSGA-II選択アルゴリズムが利用可能か確認
        assert hasattr(tools, 'selNSGA2')
        
        # テスト用の個体群を作成
        from deap import creator, base
        
        # フィットネスクラスを作成
        if hasattr(creator, "FitnessMultiTest"):
            delattr(creator, "FitnessMultiTest")
        if hasattr(creator, "IndividualTest"):
            delattr(creator, "IndividualTest")
            
        creator.create("FitnessMultiTest", base.Fitness, weights=(1.0, -1.0))
        creator.create("IndividualTest", list, fitness=creator.FitnessMultiTest)
        
        # テスト個体群を作成
        individuals = []
        for i in range(4):
            ind = creator.IndividualTest([i, i+1])
            ind.fitness.values = (i * 0.1, i * 0.05)
            individuals.append(ind)
        
        # NSGA-II選択を実行
        selected = tools.selNSGA2(individuals, 2)
        
        # 結果確認
        assert len(selected) == 2
        assert all(hasattr(ind, 'fitness') for ind in selected)

    def test_multi_objective_error_handling(self):
        """多目的最適化のエラーハンドリングテスト"""
        mock_backtest_service = Mock()
        evaluator = IndividualEvaluator(mock_backtest_service)
        
        # エラーが発生するバックテスト結果
        error_result = {
            "performance_metrics": {
                "total_return": None,  # None値
                "sharpe_ratio": float('inf'),  # 無限大
                "max_drawdown": -0.1,  # 負の値
                "total_trades": 0,  # 取引回数0
            }
        }
        
        config = GAConfig.create_multi_objective()
        
        # エラーハンドリングのテスト
        fitness_values = evaluator._calculate_multi_objective_fitness(
            error_result, config
        )
        
        # エラー時は適切なデフォルト値が返されることを確認
        assert isinstance(fitness_values, tuple)
        assert len(fitness_values) == len(config.objectives)

    def test_pareto_front_concept(self):
        """パレート最適解の概念テスト"""
        # パレート最適解の例
        solutions = [
            {"return": 0.15, "drawdown": 0.08},  # 解1
            {"return": 0.12, "drawdown": 0.05},  # 解2（低リスク）
            {"return": 0.18, "drawdown": 0.12},  # 解3（高リターン）
            {"return": 0.10, "drawdown": 0.10},  # 解4（劣解）
        ]
        
        # 解4は解2に支配される（リターンが低く、リスクが高い）
        # 解1、解2、解3はパレート最適解
        
        def is_dominated(sol1, sol2):
            """sol1がsol2に支配されるかチェック"""
            return (sol1["return"] <= sol2["return"] and 
                   sol1["drawdown"] >= sol2["drawdown"] and
                   (sol1["return"] < sol2["return"] or sol1["drawdown"] > sol2["drawdown"]))
        
        # 解4が解2に支配されることを確認
        assert is_dominated(solutions[3], solutions[1])
        
        # 解1、解2、解3は互いに支配されないことを確認
        assert not is_dominated(solutions[0], solutions[1])
        assert not is_dominated(solutions[1], solutions[0])
        assert not is_dominated(solutions[0], solutions[2])
        assert not is_dominated(solutions[2], solutions[0])


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    
    # テスト実行
    pytest.main([__file__, "-v"])
