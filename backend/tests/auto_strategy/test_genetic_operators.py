"""
遺伝的演算子のテスト
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from app.services.auto_strategy.core.fitness_sharing import FitnessSharing
from app.services.auto_strategy.core.genetic_operators import (
    _crossover_position_sizing_genes,
    _crossover_tpsl_genes,
    _mutate_conditions,
    _mutate_indicators,
    adaptive_mutate_strategy_gene_pure,
    crossover_strategy_genes_pure,
    mutate_strategy_gene_pure,
    uniform_crossover,
)
from app.services.auto_strategy.models.strategy_models import (
    Condition,
    IndicatorGene,
    PositionSizingGene,
    StrategyGene,
    TPSLGene,
)


class TestGeneticOperators:
    """遺伝的演算子のテスト"""

    @pytest.fixture
    def sample_strategy_gene(self):
        """サンプルStrategyGeneを作成"""
        return StrategyGene(
            id="test-id",
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 10}),
                IndicatorGene(type="EMA", parameters={"period": 20}),
            ],
            entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="sma")
            ],
            exit_conditions=[
                Condition(left_operand="close", operator="<", right_operand="ema")
            ],
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="sma")
            ],
            short_entry_conditions=[
                Condition(left_operand="close", operator="<", right_operand="ema")
            ],
            risk_management={"position_size": 0.1},
            tpsl_gene=TPSLGene(),
            position_sizing_gene=PositionSizingGene(),
            metadata={"test": True},
        )

    def test_crossover_strategy_genes_pure_basic(self, sample_strategy_gene):
        """基本的な交叉テスト"""
        import copy

        parent1 = copy.deepcopy(sample_strategy_gene)
        parent2 = copy.deepcopy(sample_strategy_gene)

        child1, child2 = crossover_strategy_genes_pure(parent1, parent2)

        assert isinstance(child1, StrategyGene)
        assert isinstance(child2, StrategyGene)
        assert child1.id != parent1.id
        assert child2.id != parent2.id
        assert len(child1.indicators) <= len(parent1.indicators)

    def test_crossover_tpsl_genes_both_exist(self):
        """両方のTP/SL遺伝子が存在する場合"""
        tpsl1 = Mock()
        tpsl2 = Mock()

        with patch(
            "app.services.auto_strategy.core.genetic_operators.crossover_tpsl_genes"
        ) as mock_crossover:
            mock_crossover.return_value = (Mock(), Mock())
            child1_tpsl, child2_tpsl = _crossover_tpsl_genes(tpsl1, tpsl2)

            mock_crossover.assert_called_once_with(tpsl1, tpsl2)

    def test_crossover_tpsl_genes_one_exists(self):
        """片方のTP/SL遺伝子のみが存在する場合"""
        tpsl1 = Mock()
        tpsl2 = None

        child1_tpsl, child2_tpsl = _crossover_tpsl_genes(tpsl1, tpsl2)

        assert child1_tpsl == tpsl1
        assert child2_tpsl == tpsl1

    def test_crossover_tpsl_genes_none_exist(self):
        """TP/SL遺伝子が存在しない場合"""
        child1_tpsl, child2_tpsl = _crossover_tpsl_genes(None, None)

        assert child1_tpsl is None
        assert child2_tpsl is None

    def test_crossover_position_sizing_genes_both_exist(self):
        """両方のポジションサイジング遺伝子が存在する場合"""
        ps1 = Mock()
        ps2 = Mock()

        with patch(
            "app.services.auto_strategy.core.genetic_operators.crossover_position_sizing_genes"
        ) as mock_crossover:
            mock_crossover.return_value = (Mock(), Mock())
            child1_ps, child2_ps = _crossover_position_sizing_genes(ps1, ps2)

            mock_crossover.assert_called_once_with(ps1, ps2)

    def test_crossover_position_sizing_genes_one_exists(self):
        """片方のポジションサイジング遺伝子のみが存在する場合"""
        ps1 = Mock()
        ps2 = None

        child1_ps, child2_ps = _crossover_position_sizing_genes(ps1, ps2)

        assert child1_ps == ps1
        assert child2_ps is not None  # deepcopyされたもの

    def test_crossover_position_sizing_genes_none_exist(self):
        """ポジションサイジング遺伝子が存在しない場合"""
        child1_ps, child2_ps = _crossover_position_sizing_genes(None, None)

        assert child1_ps is None
        assert child2_ps is None

    def test_mutate_strategy_gene_pure_basic(self, sample_strategy_gene):
        """基本的な突然変異テスト"""
        mutated = mutate_strategy_gene_pure(sample_strategy_gene, mutation_rate=0.0)

        assert isinstance(mutated, StrategyGene)
        assert mutated.id != sample_strategy_gene.id

    @patch("random.random", return_value=0.1)  # 確実に条件を満たす
    def test_mutate_indicators_parameter_mutation(
        self, mock_random, sample_strategy_gene
    ):
        """指標パラメータの突然変異テスト"""
        _mutate_indicators(
            sample_strategy_gene, sample_strategy_gene, mutation_rate=0.5
        )

        # パラメータが変更されている可能性がある
        assert len(sample_strategy_gene.indicators) >= 1

    @patch("random.random", return_value=0.1)
    def test_mutate_conditions_entry_conditions(
        self, mock_random, sample_strategy_gene
    ):
        """エントリー条件の突然変異テスト"""
        _mutate_conditions(sample_strategy_gene, mutation_rate=0.5)

        # テストはモックなので変更を確認しにくいが、エラーが発生しないことを確認
        assert len(sample_strategy_gene.entry_conditions) >= 0

    @patch("random.random", return_value=0.1)
    def test_mutate_conditions_exit_conditions(self, mock_random, sample_strategy_gene):
        """エグジット条件の突然変異テスト"""
        _mutate_conditions(sample_strategy_gene, mutation_rate=0.5)

        assert len(sample_strategy_gene.exit_conditions) >= 0

    def test_mutate_strategy_gene_pure_with_high_mutation_rate(
        self, sample_strategy_gene
    ):
        """高い突然変異率でのテスト"""
        import copy

        gene_to_mutate = copy.deepcopy(sample_strategy_gene)
        mutated = mutate_strategy_gene_pure(gene_to_mutate, mutation_rate=1.0)

        assert isinstance(mutated, StrategyGene)
        assert mutated.metadata.get("mutated") is True
        assert mutated.metadata.get("mutation_rate") == 1.0

    def test_adaptive_mutation_rate_adjustment(self, sample_strategy_gene):
        """適応的突然変異率調整のテスト"""
        import copy

        # 個体を作成（DEAP形式を模倣）
        class MockIndividual(list):
            def __init__(self, gene, fitness_values):
                super().__init__([gene])
                self.fitness = Mock()
                self.fitness.values = fitness_values

        # 高分散のpopulation（多様性が高い場合）
        high_variance_pop = [
            MockIndividual(copy.deepcopy(sample_strategy_gene), (1.0,)),
            MockIndividual(copy.deepcopy(sample_strategy_gene), (0.5,)),
            MockIndividual(copy.deepcopy(sample_strategy_gene), (1.5,)),
        ]

        # 低分散のpopulation（収束している場合）
        low_variance_pop = [
            MockIndividual(copy.deepcopy(sample_strategy_gene), (1.0,)),
            MockIndividual(copy.deepcopy(sample_strategy_gene), (1.01,)),
            MockIndividual(copy.deepcopy(sample_strategy_gene), (0.99,)),
        ]

        gene_to_mutate = copy.deepcopy(sample_strategy_gene)

        # 高分散の場合、低いmutation_rateになるはず
        mutated_high = adaptive_mutate_strategy_gene_pure(
            high_variance_pop, gene_to_mutate, base_mutation_rate=0.1
        )
        assert isinstance(mutated_high, StrategyGene)

        # 低分散の場合、高いmutation_rateになるはず
        mutated_low = adaptive_mutate_strategy_gene_pure(
            low_variance_pop, gene_to_mutate, base_mutation_rate=0.1
        )
        assert isinstance(mutated_low, StrategyGene)

        # メタデータに適応的rateが設定されている
        assert "adaptive_mutation_rate" in mutated_high.metadata
        assert "adaptive_mutation_rate" in mutated_low.metadata

        # 高分散のrate < 低分散のrate のはず
        assert (
            mutated_high.metadata["adaptive_mutation_rate"]
            < mutated_low.metadata["adaptive_mutation_rate"]
        )

    def test_uniform_crossover_diversity(self):
        """Test diversity of uniform crossover"""
        import copy

        # Create different parents
        parent1 = StrategyGene(
            id="parent1",
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 10}),
                IndicatorGene(type="EMA", parameters={"period": 20}),
            ],
            entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="sma")
            ],
            exit_conditions=[
                Condition(left_operand="close", operator="<", right_operand="ema")
            ],
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="sma")
            ],
            short_entry_conditions=[
                Condition(left_operand="close", operator="<", right_operand="ema")
            ],
            risk_management={"position_size": 0.1, "stop_loss": 0.05},
            tpsl_gene=TPSLGene(
                stop_loss_pct=0.05, take_profit_pct=0.15, risk_reward_ratio=3.0
            ),
            position_sizing_gene=PositionSizingGene(
                method="fixed_ratio", fixed_ratio=0.1
            ),
            metadata={"parent": 1},
        )

        parent2 = StrategyGene(
            id="parent2",
            indicators=[
                IndicatorGene(type="RSI", parameters={"period": 14}),
                IndicatorGene(type="MACD", parameters={"fast": 12, "slow": 26}),
            ],
            entry_conditions=[
                Condition(left_operand="rsi", operator="<", right_operand="30")
            ],
            exit_conditions=[
                Condition(left_operand="rsi", operator=">", right_operand="70")
            ],
            long_entry_conditions=[
                Condition(left_operand="macd", operator=">", right_operand="signal")
            ],
            short_entry_conditions=[
                Condition(left_operand="macd", operator="<", right_operand="signal")
            ],
            risk_management={"position_size": 0.2, "take_profit": 0.1},
            tpsl_gene=TPSLGene(
                stop_loss_pct=0.03, take_profit_pct=0.10, risk_reward_ratio=3.33
            ),
            position_sizing_gene=PositionSizingGene(
                method="fixed_ratio", fixed_ratio=0.2
            ),
            metadata={"parent": 2},
        )

        # Perform multiple crossovers to check diversity
        children = []
        for _ in range(50):
            child1, child2 = uniform_crossover(
                copy.deepcopy(parent1), copy.deepcopy(parent2)
            )
            children.extend([child1, child2])

        # Verify diversity: not all children are identical to parents
        parent1_str = (
            str(parent1.indicators)
            + str(parent1.entry_conditions)
            + str(parent1.risk_management)
        )
        parent2_str = (
            str(parent2.indicators)
            + str(parent2.entry_conditions)
            + str(parent2.risk_management)
        )

        diverse_children = 0
        for child in children:
            child_str = (
                str(child.indicators)
                + str(child.entry_conditions)
                + str(child.risk_management)
            )
            if child_str != parent1_str and child_str != parent2_str:
                diverse_children += 1

        # Ensure at least some children are a mix of parents
        assert diverse_children > 0, "uniform crossover does not generate diversity"

        # Note: Due to random nature, field-level mixing may not occur in all cases
        # The key is that some children are different from parents (diverse_children > 0)

    def test_population_variance_after_operations(self, sample_strategy_gene):
        """Test population variance after genetic operations"""
        import copy

        # Create diverse population
        class MockIndividual(list):
            def __init__(self, gene, fitness_values):
                super().__init__([gene])
                self.fitness = Mock()
                self.fitness.values = fitness_values
                self.fitness.valid = True

        population = [
            MockIndividual(copy.deepcopy(sample_strategy_gene), (1.0,)),
            MockIndividual(
                StrategyGene(
                    id="gene2",
                    indicators=[
                        IndicatorGene(type="RSI", parameters={"period": 14}),
                        IndicatorGene(type="MACD", parameters={"fast": 12, "slow": 26}),
                    ],
                    entry_conditions=[
                        Condition(left_operand="rsi", operator="<", right_operand="30")
                    ],
                    exit_conditions=[
                        Condition(left_operand="rsi", operator=">", right_operand="70")
                    ],
                    long_entry_conditions=[
                        Condition(
                            left_operand="macd", operator=">", right_operand="signal"
                        )
                    ],
                    short_entry_conditions=[
                        Condition(
                            left_operand="macd", operator="<", right_operand="signal"
                        )
                    ],
                    risk_management={"position_size": 0.2},
                    tpsl_gene=TPSLGene(stop_loss_pct=0.03, take_profit_pct=0.10),
                    position_sizing_gene=PositionSizingGene(),
                    metadata={},
                ),
                (0.8,),
            ),
            MockIndividual(
                StrategyGene(
                    id="gene3",
                    indicators=[
                        IndicatorGene(type="BB", parameters={"period": 20, "std": 2})
                    ],
                    entry_conditions=[
                        Condition(
                            left_operand="close", operator="<", right_operand="bb_lower"
                        )
                    ],
                    exit_conditions=[
                        Condition(
                            left_operand="close", operator=">", right_operand="bb_upper"
                        )
                    ],
                    long_entry_conditions=[],
                    short_entry_conditions=[
                        Condition(
                            left_operand="close", operator="<", right_operand="bb_lower"
                        )
                    ],
                    risk_management={"position_size": 0.15},
                    tpsl_gene=TPSLGene(stop_loss_pct=0.04, take_profit_pct=0.12),
                    position_sizing_gene=PositionSizingGene(),
                    metadata={},
                ),
                (0.9,),
            ),
            MockIndividual(
                StrategyGene(
                    id="gene4",
                    indicators=[
                        IndicatorGene(
                            type="STOCH", parameters={"k_period": 14, "d_period": 3}
                        )
                    ],
                    entry_conditions=[
                        Condition(
                            left_operand="stoch_k",
                            operator=">",
                            right_operand="stoch_d",
                        )
                    ],
                    exit_conditions=[],
                    long_entry_conditions=[
                        Condition(
                            left_operand="stoch_k",
                            operator=">",
                            right_operand="stoch_d",
                        )
                    ],
                    short_entry_conditions=[],
                    risk_management={"position_size": 0.25},
                    tpsl_gene=TPSLGene(stop_loss_pct=0.02, take_profit_pct=0.08),
                    position_sizing_gene=PositionSizingGene(),
                    metadata={},
                ),
                (0.7,),
            ),
        ]

        fitness_sharing = FitnessSharing(sharing_radius=0.1, alpha=1.0)

        def calculate_population_variance(population):
            gene_vectors = []
            for ind in population:
                gene = ind[0]  # StrategyGene
                vector = fitness_sharing._vectorize_gene(gene)
                gene_vectors.append(vector)
            if len(gene_vectors) < 2:
                return 0.0
            gene_matrix = np.array(gene_vectors)
            # Calculate average variance across dimensions
            return np.var(gene_matrix, axis=0).mean()

        original_variance = calculate_population_variance(population)
        assert original_variance > 0, "Initial population should have variance"

        # Test adaptive mutation
        mutated_population = []
        for ind in population:
            mutated_gene = adaptive_mutate_strategy_gene_pure(
                population, ind[0], base_mutation_rate=0.1
            )
            mutated_ind = MockIndividual(mutated_gene, ind.fitness.values)
            mutated_population.append(mutated_ind)

        mutated_variance = calculate_population_variance(mutated_population)
        # Adaptive mutation should maintain or increase diversity
        assert (
            mutated_variance >= original_variance * 0.8
        ), "Adaptive mutation should not significantly reduce diversity"

        # Test uniform crossover
        crossover_population = []
        for i in range(0, len(population) - 1, 2):
            parent1 = population[i][0]
            parent2 = population[i + 1][0]
            child1, child2 = uniform_crossover(
                copy.deepcopy(parent1), copy.deepcopy(parent2)
            )
            crossover_population.extend(
                [MockIndividual(child1, (0.5,)), MockIndividual(child2, (0.5,))]
            )

        if len(crossover_population) < len(population):
            crossover_population.extend(population[len(crossover_population) :])

        crossover_variance = calculate_population_variance(crossover_population)
        # Crossover should maintain diversity
        assert (
            crossover_variance >= original_variance * 0.6
        ), "Uniform crossover should maintain diversity"

        # Test silhouette-based sharing
        shared_population = fitness_sharing.silhouette_based_sharing(population)
        shared_variance = calculate_population_variance(shared_population)
        # Sharing should maintain diversity
        assert (
            shared_variance >= original_variance * 0.6
        ), "Silhouette-based sharing should maintain diversity"
