"""
遺伝的演算子のテスト
"""

import copy
from unittest.mock import Mock, patch

import pytest

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.genes import (
    Condition,
    EntryGene,
    IndicatorGene,
    PositionSizingGene,
    StrategyGene,
    TPSLGene,
)
from app.services.auto_strategy.genes.strategy_operators import mutate_conditions
from app.services.auto_strategy.config.constants import (
    EntryType,
    PositionSizingMethod,
    TPSLMethod,
)


class TestGeneticOperators:
    """遺伝的演算子のテスト"""

    @pytest.fixture
    def ga_config(self):
        from app.services.auto_strategy.config import GAConfig

        return GAConfig()

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

    def test_mutate_strategy_gene_sub_gene_specs_follow_strategy_definition(
        self, ga_config
    ):
        """突然変異対象のサブ遺伝子列挙が StrategyGene 定義に追従することを確認"""
        from app.services.auto_strategy.genes.strategy_operators import (
            _iter_mutable_sub_gene_specs,
        )

        specs = list(_iter_mutable_sub_gene_specs(ga_config))
        field_names = [field_name for field_name, _, _ in specs]
        multipliers = {
            field_name: creation_prob for field_name, _, creation_prob in specs
        }

        assert field_names == [
            "tpsl_gene",
            "long_tpsl_gene",
            "short_tpsl_gene",
            "position_sizing_gene",
            "entry_gene",
            "long_entry_gene",
            "short_entry_gene",
            "exit_gene",
        ]
        assert (
            multipliers["tpsl_gene"]
            == ga_config.mutation_config.tpsl_gene_creation_multiplier
        )
        assert (
            multipliers["position_sizing_gene"]
            == ga_config.mutation_config.position_sizing_gene_creation_multiplier
        )
        assert multipliers["entry_gene"] == 0.2
        assert multipliers["long_entry_gene"] == 0.2
        assert multipliers["short_entry_gene"] == 0.2
        assert multipliers["exit_gene"] == 0.2

    def test_mutate_strategy_gene_mutates_existing_entry_genes(self, ga_config):
        """既存の entry_gene 系が突然変異対象に含まれることを確認"""
        gene = StrategyGene(
            entry_gene=EntryGene(entry_type=EntryType.MARKET),
            long_entry_gene=EntryGene(entry_type=EntryType.LIMIT),
            short_entry_gene=EntryGene(entry_type=EntryType.STOP),
        )
        mutated_entry = EntryGene(entry_type=EntryType.LIMIT, limit_offset_pct=0.01)
        mutated_long = EntryGene(entry_type=EntryType.STOP, stop_offset_pct=0.02)
        mutated_short = EntryGene(
            entry_type=EntryType.STOP_LIMIT,
            order_validity_bars=12,
        )

        with patch(
            "app.services.auto_strategy.genes.entry.EntryGene.mutate",
            autospec=True,
            side_effect=[mutated_entry, mutated_long, mutated_short],
        ) as mutate_mock:
            mutated = gene.mutate(ga_config, mutation_rate=1.0)

        assert mutate_mock.call_count == 3
        assert mutated.entry_gene is mutated_entry
        assert mutated.long_entry_gene is mutated_long
        assert mutated.short_entry_gene is mutated_short

    def test_mutate_strategy_gene_creates_tpsl_gene_using_nested_config(self):
        """MutationConfig の nested 設定と TP/SL 制約が新規生成にも反映されることを確認"""
        config = GAConfig(
            mutation_config={
                "tpsl_gene_creation_multiplier": 1.0,
                "position_sizing_gene_creation_multiplier": 0.0,
            },
            tpsl_method_constraints=["risk_reward_ratio"],
        )
        gene = StrategyGene(
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="open")
            ],
            short_entry_conditions=[
                Condition(left_operand="close", operator="<", right_operand="open")
            ],
        )

        with patch("random.random", return_value=0.0), patch(
            "random.choice", side_effect=lambda seq: seq[0]
        ):
            mutated = gene.mutate(config, mutation_rate=1.0)

        assert mutated.tpsl_gene is not None
        assert mutated.tpsl_gene.method == TPSLMethod.RISK_REWARD_RATIO

    def test_mutate_strategy_gene_respects_position_sizing_config_ranges(self):
        """既存の PositionSizingGene 突然変異も config の探索レンジを使う"""
        config = GAConfig(
            position_sizing_method_constraints=["fixed_quantity"],
            position_sizing_fixed_ratio_range=[0.2, 0.21],
            position_sizing_fixed_quantity_range=[2.0, 2.1],
            position_sizing_max_size_range=[20.0, 21.0],
            position_sizing_var_confidence_range=[0.9, 0.91],
            position_sizing_var_lookback_range=[60, 61],
        )
        gene = StrategyGene(
            position_sizing_gene=PositionSizingGene(
                method=PositionSizingMethod.FIXED_RATIO,
                fixed_ratio=0.1,
                fixed_quantity=1.0,
                max_position_size=10.0,
                var_confidence=0.95,
                var_lookback=100,
            ),
        )

        with patch("random.random", return_value=0.0), patch(
            "random.uniform", return_value=1.5
        ):
            mutated = gene.mutate(config, mutation_rate=1.0)

        assert mutated.position_sizing_gene is not None
        assert mutated.position_sizing_gene.method == PositionSizingMethod.FIXED_QUANTITY
        assert mutated.position_sizing_gene.fixed_ratio == pytest.approx(0.2)
        assert mutated.position_sizing_gene.fixed_quantity == pytest.approx(2.0)
        assert mutated.position_sizing_gene.max_position_size == pytest.approx(20.0)
        assert mutated.position_sizing_gene.var_confidence == pytest.approx(0.91)
        assert mutated.position_sizing_gene.var_lookback == 61

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
                copy.deepcopy(parent1),
                copy.deepcopy(parent2),
                ga_config,
                crossover_type="uniform",
            )
            children.extend([child1, child2])

        diverse = False
        for child in children:
            # 親と完全に同じでない子がいればOK
            if len(child.indicators) > 0 and (
                child.indicators[0].type != parent1.indicators[0].type
                or child.risk_management["position_size"]
                != parent1.risk_management["position_size"]
            ):
                # parent1と違う
                if len(child.indicators) > 0 and (
                    child.indicators[0].type != parent2.indicators[0].type
                    or child.risk_management["position_size"]
                    != parent2.risk_management["position_size"]
                ):
                    # parent2とも違う -> 混ざっている
                    diverse = True
                    break

        # 確率的なので失敗する可能性もゼロではないが、50回ならほぼ確実に混ざる
        assert diverse, "Uniform crossover should generate diverse offspring"

    def test_mutate_conditions_includes_exit_condition_branches(self, ga_config):
        """exit 条件も mutate_conditions の対象に含まれることを確認"""
        gene = StrategyGene(
            long_exit_conditions=[
                Condition(left_operand="close", operator=">", right_operand=110.0)
            ],
            short_exit_conditions=[
                Condition(left_operand="close", operator="<", right_operand=90.0)
            ],
        )

        with patch("random.random", return_value=0.0), patch(
            "random.randint", return_value=0
        ), patch("random.choice", return_value="!="):
            mutate_conditions(gene, 1.0, ga_config)

        assert gene.long_exit_conditions[0].operator == "!="
        assert gene.short_exit_conditions[0].operator == "!="
