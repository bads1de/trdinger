import pytest
from unittest.mock import MagicMock, patch
from app.services.auto_strategy.genes.strategy import StrategyGene
from app.services.auto_strategy.genes.indicator import IndicatorGene
from app.services.auto_strategy.genes.conditions import Condition


class TestStrategyGene:
    @pytest.fixture
    def sample_gene(self):
        return StrategyGene.create_default()

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.risk_param_mutation_range = (0.8, 1.2)
        config.indicator_param_mutation_range = (0.8, 1.2)
        config.tpsl_gene_creation_probability_multiplier = 1.0
        config.position_sizing_gene_creation_probability_multiplier = 1.0
        config.indicator_add_delete_probability = 0.5
        config.indicator_add_vs_delete_probability = 0.5
        config.max_indicators = 5
        config.min_indicators = 1
        config.condition_change_probability_multiplier = 1.0
        config.condition_selection_probability = 1.0
        config.condition_operator_switch_probability = 0.5
        config.valid_condition_operators = [">", "<", "=="]
        config.crossover_field_selection_probability = 0.5
        config.adaptive_mutation_variance_threshold = 0.01
        config.adaptive_mutation_rate_decrease_multiplier = 0.5
        config.adaptive_mutation_rate_increase_multiplier = 2.0
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

        assert cond.operator in mock_config.valid_condition_operators

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
        mock_config.adaptive_mutation_variance_threshold = 0.0001
        mutated = sample_gene.adaptive_mutate(
            population, mock_config, base_mutation_rate=0.5
        )
        assert mutated.metadata["adaptive_mutation_rate"] < 0.5

        # 分散が小さい場合 -> 変異率上昇
        mock_config.adaptive_mutation_variance_threshold = 100.0
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
