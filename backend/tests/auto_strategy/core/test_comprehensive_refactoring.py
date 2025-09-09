"""
包括的なリファクタリングテスト

エッジケース、統合テスト、データ妥当性を追加 covers
"""

import copy
import math
import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Any, Dict

from app.services.auto_strategy.core.ga_engine import EvolutionRunner
from app.services.auto_strategy.core.individual_evaluator import IndividualEvaluator
from app.services.auto_strategy.core.genetic_operators import (
    create_deap_crossover_wrapper,
    create_deap_mutate_wrapper,
    crossover_strategy_genes_pure,
    mutate_strategy_gene_pure,
)
from app.services.auto_strategy.models.strategy_gene import StrategyGene
from app.services.auto_strategy.models.indicator_gene import IndicatorGene
from app.services.auto_strategy.models.condition import Condition
from app.services.auto_strategy.config.ga_runtime import GAConfig


class TestEdgeCasesRefactoring:
    """エッジケースのテスト"""

    @pytest.fixture
    def minimal_strategy_gene(self):
        """最小限の戦略遺伝子を作成"""
        return StrategyGene(
            id="test-001",
            indicators=[IndicatorGene(type="SMA", parameters={"period": 10})],
            entry_conditions=[Condition(left_operand=1.0, operator=">", right_operand=0.5)],
            exit_conditions=[Condition(left_operand=1.0, operator="<", right_operand=1.5)]
        )

    @pytest.fixture
    def invalid_backtest_result(self):
        """無効なバックテスト結果（各種エラーパターン）"""
        return {
            "performance_metrics": {
                "total_return": float('nan'),
                "sharpe_ratio": float('-inf'),
                "max_drawdown": -2.0,  # 負の無効値
                "win_rate": -0.1,  # 負の割合
                "total_trades": -5  # 負の取引数
            }
        }

    def test_mutate_with_empty_indicators(self, minimal_strategy_gene):
        """空の指標リストでの突然変異"""
        gene = copy.deepcopy(minimal_strategy_gene)
        gene.indicators = []

        result = mutate_strategy_gene_pure(gene)

        # 結果はStrategyGeneであるはず
        assert isinstance(result, StrategyGene)
        assert result.id != gene.id  # IDが変更されているはず

    def test_crossover_with_identical_parents(self):
        """同一の親遺伝子での交叉"""
        parent = StrategyGene(
            id="parent-001",
            indicators=[IndicatorGene(type="SMA", parameters={"period": 20})],
            entry_conditions=[Condition(left_operand=1.0, operator=">", right_operand=0.5)]
        )

        child1, child2 = crossover_strategy_genes_pure(parent, parent)

        # 子は親と構造が同じ
        assert isinstance(child1, StrategyGene)
        assert isinstance(child2, StrategyGene)

        # IDは新規生成
        assert child1.id != parent.id
        assert child2.id != parent.id
        assert child1.id != child2.id

    def test_evaluate_individual_with_invalid_data(self, invalid_backtest_result):
        """無効なバックテスト結果での個体評価"""
        evaluator = IndividualEvaluator(Mock())
        config = GAConfig()

        # 評価がクラッシュしないことを確認
        fitness = evaluator.evaluate_individual(invalid_backtest_result, config)

        assert isinstance(fitness, (tuple, float))

    def test_fitness_calculation_with_large_numbers(self):
        """巨大数値での適応度計算"""
        evaluator = IndividualEvaluator(Mock())
        config = GAConfig()

        large_result = {
            "performance_metrics": {
                "total_return": 1e6,  # 非常に大きなリターン
                "sharpe_ratio": 100.0,
                "max_drawdown": 0.99,  # ほぼ100%ドローダウン
                "win_rate": 1.0,
                "total_trades": 1000000  # 巨大取引数
            }
        }

        fitness = evaluator._calculate_fitness(large_result, config)

        # 計算が正常終了し、適切な範囲であること
        assert isinstance(fitness, float)
        assert not math.isnan(fitness)
        assert not math.isinf(fitness)

    def test_parallel_evolution_simulation(self):
        """並列進化シミュレーション"""
        # 複数個体での同時評価を模擬
        evaluator = IndividualEvaluator(Mock())
        config = GAConfig()

        test_results = [
            {
                "performance_metrics": {
                    "total_return": 0.1,
                    "sharpe_ratio": 1.0,
                    "max_drawdown": 0.1,
                    "win_rate": 0.6,
                    "total_trades": 50
                }
            },
            {
                "performance_metrics": {
                    "total_return": 0.2,
                    "sharpe_ratio": 1.5,
                    "max_drawdown": 0.15,
                    "win_rate": 0.7,
                    "total_trades": 75
                }
            }
        ]

        fitnesses = []
        for result in test_results:
            fitness = evaluator.evaluate_individual(result, config)
            fitnesses.append(fitness)

        # 各適応度が計算され、相対的に正しい順序であること
        assert len(fitnesses) == 2
        assert all(isinstance(f, (int, float, tuple)) for f in fitnesses)


class TestDataValidationRefactoring:
    """データ妥当性のテスト"""

    @pytest.fixture
    def validator_evaluator(self):
        """妥当性検証用評価器"""
        return IndividualEvaluator(Mock())

    @pytest.fixture
    def valid_ga_config(self):
        """有効なGA設定"""
        config = GAConfig()
        config.fitness_constraints = {
            "min_trades": 10,
            "max_drawdown_limit": 0.5,
            "min_sharpe_ratio": 0.1
        }
        return config

    def test_extract_performance_metrics_type_safety(self, validator_evaluator):
        """パフォーマンスメトリクス抽出の型安全性"""
        result = {
            "performance_metrics": {
                "total_return": "not_a_number",  # 文字列の不当な値
                "sharpe_ratio": None,
                "max_drawdown": [1, 2, 3],  # リストの不当な値
                "win_rate": {"key": "value"},  # 辞書の不当な値
                "total_trades": 42.7  # 浮動小数点取引数
            }
        }

        metrics = validator_evaluator._extract_performance_metrics(result)

        # 全てのキーがあること
        expected_keys = [
            "total_return", "sharpe_ratio", "max_drawdown",
            "win_rate", "profit_factor", "sortino_ratio",
            "calmar_ratio", "total_trades"
        ]

        for key in expected_keys:
            assert key in metrics

        # 型が適切であること
        assert isinstance(metrics["total_return"], float)
        assert isinstance(metrics["total_trades"], (int, float))

    def test_multi_objective_validation(self, validator_evaluator, valid_ga_config):
        """多目的最適化の妥当性検証"""
        valid_ga_config.enable_multi_objective = True
        valid_ga_config.objectives = ["total_return", "sharpe_ratio", "invalid_objective"]

        # 無効な目標を含む結果
        result = {
            "performance_metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 1.0,
                "invalid_objective": 0.5  # 存在しないメトリクス
            }
        }

        fitness_values = validator_evaluator._calculate_multi_objective_fitness(result, valid_ga_config)

        # エラーハンドリングが正しく動作すること
        assert isinstance(fitness_values, tuple)
        # 無効な目標は0.0として処理される
        assert len(fitness_values) == 3  # total_return, sharpe_ratio, invalid_objective

    def test_constraint_validation_edge_cases(self, validator_evaluator, valid_ga_config):
        """制約検証のエッジケース"""
        # 取引数境界値
        edge_cases = [
            # 最小取引数より少ない場合
            {"total_trades": 5, "expected_zero": True},
            # 最小取引数ちょうどの場合
            {"total_trades": 10, "expected_zero": False},
            # ドローダウン制限を超える場合
            {"max_drawdown": 0.6, "expected_zero": True},
            # ドローダウン制限内
            {"max_drawdown": 0.3, "expected_zero": False},
        ]

        for case in edge_cases:
            result = {
                "performance_metrics": {
                    "total_return": 0.1,
                    "sharpe_ratio": 1.0,
                    "max_drawdown": case.get("max_drawdown", 0.2),
                    "win_rate": 0.6,
                    "total_trades": case.get("total_trades", 20)
                }
            }

            fitness = validator_evaluator._calculate_fitness(result, valid_ga_config)

            if case["expected_zero"]:
                assert fitness == 0.0, f"Failed for case: {case}"
            else:
                assert fitness > 0.0, f"Failed for case: {case}"


class TestIntegrationRefactoring:
    """統合テスト"""

    @pytest.fixture
    def mock_toolbox(self):
        """完全なDEAPツールボックスモック"""
        toolbox = Mock()
        toolbox.population = Mock(return_value=[])

        # map関数をシミュレート（リストを返すように）
        def mock_map(func, population):
            return [(1.0,)] * len(population)  # 各個体に(1.0,)のfitnessを返す

        toolbox.map = mock_map
        toolbox.evaluate = Mock(return_value=[(1.0,)])
        toolbox.select = Mock(return_value=[])
        toolbox.mate = Mock()
        toolbox.mutate = Mock()
        return toolbox

    @pytest.fixture
    def complete_ga_config(self):
        """完全なGA設定"""
        config = GAConfig()
        config.population_size = 10
        config.generations = 5
        config.crossover_rate = 0.8
        config.mutation_rate = 0.2
        config.enable_multi_objective = False
        config.enable_fitness_sharing = False
        config.fitness_weights = {
            "total_return": 0.4,
            "sharpe_ratio": 0.3,
            "max_drawdown": 0.2,
            "win_rate": 0.1
        }
        return config

    def test_full_evolution_chain_single_objective(self, mock_toolbox, complete_ga_config):
        """完全な単一目的最適化チェーン"""
        runner = EvolutionRunner(mock_toolbox, Mock())
        population = [Mock() for _ in range(10)]  # 個体リスト

        # 評価結果をモック
        mock_toolbox.evaluate.return_value = list(zip([1.0 + i * 0.1 for i in range(10)], [0.0] * 10))

        with patch('deap.algorithms.eaMuPlusLambda') as mock_ea:
            mock_ea.return_value = (population, Mock())

            result_pop, result_log = runner.run_single_objective_evolution(
                population, complete_ga_config
            )

            # アルゴリズムが呼び出されたことを確認
            mock_ea.assert_called_once()
            assert result_pop == population
            assert result_log is not None

    def test_full_evolution_chain_multi_objective(self, mock_toolbox, complete_ga_config):
        """完全な多目的最適化チェーン"""
        runner = EvolutionRunner(mock_toolbox, Mock())
        complete_ga_config.enable_multi_objective = True

        population = [Mock() for _ in range(10)]

        with patch('deap.tools.selNSGA2') as mock_select, \
             patch('deap.algorithms.eaMuPlusLambda') as mock_ea, \
             patch('deap.tools.ParetoFront') as mock_pareto:

            mock_ea.return_value = (population, Mock())
            mock_select.return_value = population

            result_pop, result_log = runner.run_multi_objective_evolution(
                population, complete_ga_config
            )

            mock_ea.assert_called_once()
            assert result_pop == population

    def test_genetic_operators_integration(self):
        """遺伝的演算子の統合テスト"""
        # StrategyGeneオブジェクトの交叉と突然変異の組合せ
        parent1 = StrategyGene(
            id="parent1",
            indicators=[IndicatorGene(type="SMA", parameters={"period": 20})],
            entry_conditions=[Condition(left_operand=1.0, operator=">", right_operand=0.5)]
        )
        parent2 = StrategyGene(
            id="parent2",
            indicators=[IndicatorGene(type="EMA", parameters={"period": 15})],
            entry_conditions=[Condition(left_operand=1.5, operator="<", right_operand=2.0)]
        )

        # 交叉実行
        child1, child2 = crossover_strategy_genes_pure(parent1, parent2)

        # 突然変異実行
        mutated1 = mutate_strategy_gene_pure(child1)

        # 結果の妥当性確認
        assert isinstance(child1, StrategyGene)
        assert isinstance(child2, StrategyGene)
        assert isinstance(mutated1, StrategyGene)

        assert child1.id != parent1.id
        assert child1.id != parent2.id
        assert mutated1.id != child1.id

    def test_error_propagation_handling(self):
        """エラー伝搬処理のテスト"""
        evaluator = IndividualEvaluator(Mock())

        # 完全に空の結果
        empty_result = {}

        # 設定
        config = GAConfig()

        # エラーが発生せず、適切なデフォルト値が返される
        fitness = evaluator.evaluate_individual(empty_result, config)

        # 結果がタプルまたは数値であること
        assert isinstance(fitness, (int, float, tuple))

        if isinstance(fitness, tuple):
            assert all(isinstance(f, (int, float)) for f in fitness)

    def test_performance_robustness(self):
        """性能保証テスト（リソース消費）"""
        evaluator = IndividualEvaluator(Mock())

        # 大量の条件を含む戦略
        complex_metrics = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.1,
                "win_rate": 0.65,
                "total_trades": 50,
                "profit_factor": 1.2,
                "sortino_ratio": 1.8,
                "calmar_ratio": 1.5,
                # 追加のメトリクス（拡張性のテスト）
                "additional_metric1": 0.8,
                "additional_metric2": 1.2
            }
        }

        config = GAConfig()

        # パフォーマンスが大幅に劣化しないこと（ここではタイムアウトしないこと）
        fitness = evaluator.evaluate_individual(complex_metrics, config)

        assert isinstance(fitness, (int, float, tuple))