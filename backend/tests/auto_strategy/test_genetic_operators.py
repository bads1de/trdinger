"""
遺伝的演算子のテスト
"""

from unittest.mock import Mock, patch
import pytest
import copy

from app.services.auto_strategy.genes import (
    Condition,
    IndicatorGene,
    PositionSizingGene,
    StrategyGene,
    TPSLGene,
)

class TestGeneticOperators:
    """遺伝的演算子のテスト"""

    @pytest.fixture
    def ga_config(self):
        from app.services.auto_strategy.config import GASettings
        return GASettings()

    @pytest.fixture
    def sample_strategy_gene(self):
        """サンプルStrategyGeneを作成"""
        return StrategyGene(
            id="test-id",
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 10}),
                IndicatorGene(type="EMA", parameters={"period": 20}),
            ],
            risk_management={"position_size": 0.1},
            tpsl_gene=TPSLGene(),
            position_sizing_gene=PositionSizingGene(),
            metadata={"test": True},
        )

    def test_crossover_basic(self, sample_strategy_gene, ga_config):
        """基本的な交叉テスト"""
        parent1 = copy.deepcopy(sample_strategy_gene)
        parent2 = copy.deepcopy(sample_strategy_gene)

        child1, child2 = StrategyGene.crossover(parent1, parent2, ga_config)

        assert isinstance(child1, StrategyGene)
        assert isinstance(child2, StrategyGene)
        assert child1.id != parent1.id
        assert child2.id != parent2.id
        assert len(child1.indicators) <= len(parent1.indicators)

    def test_mutate_basic(self, sample_strategy_gene, ga_config):
        """基本的な突然変異テスト"""
        mutated = sample_strategy_gene.mutate(ga_config, mutation_rate=0.0)

        assert isinstance(mutated, StrategyGene)
        assert mutated.id != sample_strategy_gene.id

    def test_adaptive_mutate(self, sample_strategy_gene, ga_config):
        """適応的突然変異率調整のテスト"""
        # 個体を作成（DEAP形式を模倣）
        class MockIndividual:
            def __init__(self, fitness_values):
                self.fitness = Mock()
                self.fitness.values = fitness_values

        # 高分散のpopulation（多様性が高い場合）
        high_variance_pop = [
            MockIndividual((1.0,)),
            MockIndividual((0.5,)),
            MockIndividual((1.5,)),
        ]

        # 低分散のpopulation（収束している場合）
        low_variance_pop = [
            MockIndividual((1.0,)),
            MockIndividual((1.01,)),
            MockIndividual((0.99,)),
        ]

        # 高分散の場合、低いmutation_rateになるはず
        mutated_high = sample_strategy_gene.adaptive_mutate(
            high_variance_pop, ga_config, base_mutation_rate=0.1
        )
        assert isinstance(mutated_high, StrategyGene)

        # 低分散の場合、高いmutation_rateになるはず
        mutated_low = sample_strategy_gene.adaptive_mutate(
            low_variance_pop, ga_config, base_mutation_rate=0.1
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

    def test_uniform_crossover_diversity(self, ga_config):
        """ユニフォーム交叉の多様性テスト"""
        parent1 = StrategyGene(
            id="parent1",
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 10}),
            ],
            risk_management={"position_size": 0.1},
            metadata={"parent": 1},
        )

        parent2 = StrategyGene(
            id="parent2",
            indicators=[
                IndicatorGene(type="RSI", parameters={"period": 14}),
            ],
            risk_management={"position_size": 0.2},
            metadata={"parent": 2},
        )

        # 複数回実行して多様性を確認
        children = []
        for _ in range(50):
            child1, child2 = StrategyGene.crossover(
                copy.deepcopy(parent1), copy.deepcopy(parent2), ga_config, crossover_type="uniform"
            )
            children.extend([child1, child2])

        diverse = False
        for child in children:
            # 親と完全に同じでない子がいればOK
            if (len(child.indicators) > 0 and 
                (child.indicators[0].type != parent1.indicators[0].type or 
                 child.risk_management["position_size"] != parent1.risk_management["position_size"])):
                 # parent1と違う
                 if (len(child.indicators) > 0 and 
                    (child.indicators[0].type != parent2.indicators[0].type or 
                     child.risk_management["position_size"] != parent2.risk_management["position_size"])):
                     # parent2とも違う -> 混ざっている
                     diverse = True
                     break
        
        # 確率的なので失敗する可能性もゼロではないが、50回ならほぼ確実に混ざる
        assert diverse, "Uniform crossover should generate diverse offspring"