from unittest.mock import MagicMock, patch

import pytest

from app.services.auto_strategy.genes.conditions import Condition, StatefulCondition
from app.services.auto_strategy.genes.entry import EntryGene
from app.services.auto_strategy.genes.exit import ExitGene
from app.services.auto_strategy.genes.indicator import IndicatorGene
from app.services.auto_strategy.genes.position_sizing import PositionSizingGene
from app.services.auto_strategy.genes.strategy import StrategyGene
from app.services.auto_strategy.genes.tool import ToolGene
from app.services.auto_strategy.genes.tpsl import TPSLGene


class TestStrategyGene:
    @pytest.fixture
    def sample_gene(self):
        return StrategyGene.create_default()

    @pytest.fixture
    def mock_config(self):
        from app.services.auto_strategy.config import GAConfig

        config = GAConfig()
        return config

    def test_create_default(self):
        gene = StrategyGene.create_default()
        assert gene.id != ""
        assert len(gene.indicators) > 0
        assert len(gene.long_entry_conditions) > 0
        assert gene.tpsl_gene is not None
        assert gene.metadata["generated_by"] == "create_default"

    def test_assemble(self):
        indicators = [IndicatorGene(type="RSI")]
        long_cond = [Condition(left_operand="rsi", operator="<", right_operand=30)]
        short_cond = [Condition(left_operand="rsi", operator=">", right_operand=70)]

        gene = StrategyGene.assemble(
            indicators=indicators,
            long_entry_conditions=long_cond,
            short_entry_conditions=short_cond,
            metadata={"source": "test"},
        )

        assert gene.indicators == indicators
        assert gene.long_entry_conditions == long_cond
        assert gene.metadata["source"] == "test"
        assert "assembled_at" in gene.metadata

    def test_mutate_basic(self, sample_gene, mock_config):
        # 突然変異を実行（レートを高くして確実に何か起きるようにする）
        mutated = sample_gene.mutate(mock_config, mutation_rate=1.0)

        assert mutated.id != sample_gene.id
        assert mutated.metadata["mutated"] is True
        # risk_management の値が変化しているか（確率的なので検証は難しいが、呼び出されていることの確認）
        assert "position_size" in mutated.risk_management

    def test_mutate_indicators(self, sample_gene, mock_config):
        # 指標の変異のみをテスト
        sample_gene.indicators = [IndicatorGene(type="SMA", parameters={"period": 20})]

        with patch("random.random", return_value=0.0):  # 常に変異・追加
            StrategyGene._mutate_indicators(sample_gene, 1.0, mock_config)

        # パラメータが変化しているはず（シードや乱数によるが）
        # また追加・削除ロジックも通る

    def test_mutate_conditions(self, sample_gene, mock_config):
        cond = Condition(left_operand="close", operator=">", right_operand="open")
        sample_gene.long_entry_conditions = [cond]

        # 演算子の切り替えテスト
        with patch("random.random", return_value=0.0):
            StrategyGene._mutate_conditions(sample_gene, 1.0, mock_config)

        assert cond.operator in mock_config.mutation_config.valid_condition_operators

    def test_crossover_uniform(self, sample_gene, mock_config):
        parent2 = StrategyGene.create_default()
        parent2.id = "parent2"

        child1, child2 = StrategyGene.crossover(
            sample_gene, parent2, mock_config, crossover_type="uniform"
        )

        assert child1.id != sample_gene.id
        assert child1.id != parent2.id
        assert "crossover_parent1" in child1.metadata

    def test_crossover_single_point(self, sample_gene, mock_config):
        parent2 = StrategyGene.create_default()

        child1, child2 = StrategyGene.crossover(
            sample_gene, parent2, mock_config, crossover_type="single_point"
        )

        assert child1 is not None
        assert child2 is not None

    def test_adaptive_mutate(self, sample_gene, mock_config):
        # ダミーの集団
        ind1 = MagicMock()
        ind1.fitness.values = (1.0,)
        ind2 = MagicMock()
        ind2.fitness.values = (2.0,)
        population = [ind1, ind2]

        # 分散が大きい場合 -> 変異率低下
        mock_config.mutation_config.adaptive_variance_threshold = 0.0001
        mutated = sample_gene.adaptive_mutate(
            population, mock_config, base_mutation_rate=0.5
        )
        assert mutated.metadata["adaptive_mutation_rate"] < 0.5

        # 分散が小さい場合 -> 変異率上昇
        mock_config.mutation_config.adaptive_variance_threshold = 100.0
        mutated = sample_gene.adaptive_mutate(
            population, mock_config, base_mutation_rate=0.1
        )
        assert mutated.metadata["adaptive_mutation_rate"] > 0.1

    def test_properties(self, sample_gene):
        assert sample_gene.has_long_short_separation() is True

    def test_validate(self, sample_gene):
        # 実際に GeneValidator を呼び出す
        is_valid, errors = sample_gene.validate()
        # デフォルトは有効なはず
        assert is_valid is True
        assert len(errors) == 0

    def test_clone_copies_all_nested_runtime_fields(self):
        gene = StrategyGene(
            id="source-id",
            indicators=[IndicatorGene(type="SMA", parameters={"period": 20})],
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand=100.0)
            ],
            short_entry_conditions=[
                Condition(left_operand="close", operator="<", right_operand=90.0)
            ],
            stateful_conditions=[
                StatefulCondition(
                    trigger_condition=Condition(
                        left_operand="close",
                        operator=">",
                        right_operand=100.0,
                    ),
                    follow_condition=Condition(
                        left_operand="close",
                        operator="<",
                        right_operand=95.0,
                    ),
                )
            ],
            risk_management={"position_size": 0.1},
            tpsl_gene=TPSLGene(stop_loss_pct=0.02, take_profit_pct=0.04),
            long_tpsl_gene=TPSLGene(stop_loss_pct=0.03, take_profit_pct=0.05),
            short_tpsl_gene=TPSLGene(stop_loss_pct=0.04, take_profit_pct=0.06),
            position_sizing_gene=PositionSizingGene(risk_per_trade=0.02),
            entry_gene=EntryGene(),
            long_entry_gene=EntryGene(limit_offset_pct=0.01),
            short_entry_gene=EntryGene(stop_offset_pct=0.02),
            tool_genes=[ToolGene(tool_name="weekend_filter", params={"enabled": True})],
            metadata={"source": {"name": "test"}},
        )

        cloned = gene.clone()

        assert cloned.id != gene.id
        assert cloned.indicators[0] is not gene.indicators[0]
        assert cloned.long_entry_conditions[0] is not gene.long_entry_conditions[0]
        assert cloned.short_entry_conditions[0] is not gene.short_entry_conditions[0]
        assert cloned.stateful_conditions[0] is not gene.stateful_conditions[0]
        assert cloned.tpsl_gene is not gene.tpsl_gene
        assert cloned.long_tpsl_gene is not gene.long_tpsl_gene
        assert cloned.short_tpsl_gene is not gene.short_tpsl_gene
        assert cloned.position_sizing_gene is not gene.position_sizing_gene
        assert cloned.entry_gene is not gene.entry_gene
        assert cloned.long_entry_gene is not gene.long_entry_gene
        assert cloned.short_entry_gene is not gene.short_entry_gene
        assert cloned.tool_genes[0] is not gene.tool_genes[0]
        assert cloned.metadata is not gene.metadata

    def test_sub_gene_helpers_are_consistent(self):
        field_names = StrategyGene.sub_gene_field_names()
        class_map = StrategyGene.sub_gene_class_map()

        assert field_names == (
            "tpsl_gene",
            "long_tpsl_gene",
            "short_tpsl_gene",
            "position_sizing_gene",
            "entry_gene",
            "long_entry_gene",
            "short_entry_gene",
            "exit_gene",
        )
        assert set(class_map) == set(field_names)
        assert class_map["tpsl_gene"] is TPSLGene
        assert class_map["position_sizing_gene"] is PositionSizingGene
        assert class_map["entry_gene"] is EntryGene
        assert class_map["exit_gene"] is ExitGene
