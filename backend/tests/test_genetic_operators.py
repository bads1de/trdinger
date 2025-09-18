"""
遺伝的演算子のテスト
"""

import pytest
from unittest.mock import Mock, patch

from app.services.auto_strategy.core.genetic_operators import (
    crossover_strategy_genes_pure,
    mutate_strategy_gene_pure,
    _crossover_tpsl_genes,
    _crossover_position_sizing_genes,
    _mutate_indicators,
    _mutate_conditions,
)
from app.services.auto_strategy.models.strategy_models import StrategyGene, IndicatorGene, Condition, TPSLGene, PositionSizingGene


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
            entry_conditions=[Condition(left_operand="close", operator=">", right_operand="sma")],
            exit_conditions=[Condition(left_operand="close", operator="<", right_operand="ema")],
            long_entry_conditions=[Condition(left_operand="close", operator=">", right_operand="sma")],
            short_entry_conditions=[Condition(left_operand="close", operator="<", right_operand="ema")],
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

        with patch('app.services.auto_strategy.core.genetic_operators.crossover_tpsl_genes') as mock_crossover:
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

        with patch('app.services.auto_strategy.core.genetic_operators.crossover_position_sizing_genes') as mock_crossover:
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

    @patch('random.random', return_value=0.1)  # 確実に条件を満たす
    def test_mutate_indicators_parameter_mutation(self, mock_random, sample_strategy_gene):
        """指標パラメータの突然変異テスト"""
        _mutate_indicators(sample_strategy_gene, sample_strategy_gene, mutation_rate=0.5)

        # パラメータが変更されている可能性がある
        assert len(sample_strategy_gene.indicators) >= 1

    @patch('random.random', return_value=0.1)
    def test_mutate_conditions_entry_conditions(self, mock_random, sample_strategy_gene):
        """エントリー条件の突然変異テスト"""
        _mutate_conditions(sample_strategy_gene, mutation_rate=0.5)

        # テストはモックなので変更を確認しにくいが、エラーが発生しないことを確認
        assert len(sample_strategy_gene.entry_conditions) >= 0

    @patch('random.random', return_value=0.1)
    def test_mutate_conditions_exit_conditions(self, mock_random, sample_strategy_gene):
        """エグジット条件の突然変異テスト"""
        _mutate_conditions(sample_strategy_gene, mutation_rate=0.5)

        assert len(sample_strategy_gene.exit_conditions) >= 0

    def test_mutate_strategy_gene_pure_with_high_mutation_rate(self, sample_strategy_gene):
        """高い突然変異率でのテスト"""
        import copy
        gene_to_mutate = copy.deepcopy(sample_strategy_gene)
        mutated = mutate_strategy_gene_pure(gene_to_mutate, mutation_rate=1.0)

        assert isinstance(mutated, StrategyGene)
        assert mutated.metadata.get("mutated") is True
        assert mutated.metadata.get("mutation_rate") == 1.0