"""
遺伝的演算子のテスト
"""

import copy
import random
from unittest.mock import Mock, patch

import pytest

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.config.constants import (
    EntryType,
    PositionSizingMethod,
    TPSLMethod,
)
from app.services.auto_strategy.genes import (
    Condition,
    EntryGene,
    IndicatorGene,
    PositionSizingGene,
    StrategyGene,
    TPSLGene,
)
from app.services.auto_strategy.genes.operators.mutation import (
    mutate_conditions,
    mutate_risk_management_modes,
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
        assert len(child1.indicators) <= ga_config.max_indicators

    def test_mutate_basic(self, sample_strategy_gene, ga_config):
        """基本的な突然変異テスト"""
        mutated = sample_strategy_gene.mutate(ga_config, mutation_rate=0.0)

        assert isinstance(mutated, StrategyGene)
        assert mutated.id != sample_strategy_gene.id

    def test_mutate_strategy_gene_sub_gene_specs_follow_strategy_definition(
        self, ga_config
    ):
        """突然変異対象のサブ遺伝子列挙が StrategyGene 定義に追従することを確認"""
        from app.services.auto_strategy.genes.operators.mutation import (
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
        """MutationConfig の nested 設定が新規生成にも反映されることを確認"""
        config = GAConfig(
            mutation_config={
                "tpsl_gene_creation_multiplier": 1.0,
                "position_sizing_gene_creation_multiplier": 0.0,
            },
        )
        gene = StrategyGene(
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="open")
            ],
            short_entry_conditions=[
                Condition(left_operand="close", operator="<", right_operand="open")
            ],
        )

        with patch("random.random", return_value=0.0):
            mutated = gene.mutate(config, mutation_rate=1.0)

        assert mutated.tpsl_gene is not None

    def test_risk_mode_switch_changes_tpsl_method(self, ga_config):
        """TPSL method スイッチ変異は必ず別の方式へ切り替わる"""
        gene = StrategyGene(
            tpsl_gene=TPSLGene(method=TPSLMethod.FIXED_PERCENTAGE),
            long_tpsl_gene=TPSLGene(method=TPSLMethod.STATISTICAL),
            short_tpsl_gene=TPSLGene(method=TPSLMethod.ADAPTIVE),
        )
        original_methods = {
            "tpsl": gene.tpsl_gene.method,
            "long": gene.long_tpsl_gene.method,
            "short": gene.short_tpsl_gene.method,
        }

        # random.random < 0.03 を保証し、choice は決定的にする
        with patch("random.random", return_value=0.0), patch(
            "random.choice",
            side_effect=lambda seq: seq[0],
        ):
            mutate_risk_management_modes(gene, 0.1, ga_config)

        assert gene.tpsl_gene.method != original_methods["tpsl"]
        assert gene.long_tpsl_gene.method != original_methods["long"]
        assert gene.short_tpsl_gene.method != original_methods["short"]

    def test_risk_mode_switch_toggles_trailing(self, ga_config):
        """トレーリングストップ / TP のトグル変異"""
        gene = StrategyGene(
            tpsl_gene=TPSLGene(trailing_stop=False, trailing_take_profit=False),
        )
        with patch("random.random", return_value=0.0), patch(
            "random.choice", side_effect=lambda seq: seq[0]
        ):
            mutate_risk_management_modes(gene, 1.0, ga_config)

        assert gene.tpsl_gene.trailing_stop is True
        assert gene.tpsl_gene.trailing_take_profit is True

    def test_risk_mode_switch_changes_position_sizing_method(self, ga_config):
        """ポジションサイジング method スイッチ変異は必ず別方式になる"""
        gene = StrategyGene(
            position_sizing_gene=PositionSizingGene(
                method=PositionSizingMethod.VOLATILITY_BASED
            ),
        )
        with patch("random.random", return_value=0.0), patch(
            "random.choice", side_effect=lambda seq: seq[0]
        ):
            mutate_risk_management_modes(gene, 0.1, ga_config)

        assert (
            gene.position_sizing_gene.method
            != PositionSizingMethod.VOLATILITY_BASED
        )

    def test_risk_mode_switch_no_op_at_zero_rate(self, ga_config):
        """mutation_rate=0 では何も変わらない"""
        gene = StrategyGene(
            tpsl_gene=TPSLGene(method=TPSLMethod.FIXED_PERCENTAGE),
            position_sizing_gene=PositionSizingGene(),
        )
        before_tpsl = gene.tpsl_gene.method
        before_sizing = gene.position_sizing_gene.method

        mutate_risk_management_modes(gene, 0.0, ga_config)

        assert gene.tpsl_gene.method == before_tpsl
        assert gene.position_sizing_gene.method == before_sizing

    def test_mutate_strategy_gene_applies_risk_mode_switch(self, ga_config):
        """mutate_strategy_gene 全体フローで方式スイッチが機能する"""
        methods_seen: set[TPSLMethod] = set()
        gene = StrategyGene(
            indicators=[IndicatorGene(type="SMA", parameters={"period": 10})],
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="open")
            ],
            tpsl_gene=TPSLGene(method=TPSLMethod.FIXED_PERCENTAGE),
        )
        for seed in range(50):
            rng = random.Random(seed)
            with patch("random.random", rng.random), patch(
                "random.uniform", rng.uniform
            ), patch(
                "random.randint", lambda a, b, _r=rng: _r.randint(a, b)
            ), patch(
                "random.choice", lambda seq, _r=rng: _r.choice(list(seq))
            ), patch(
                "random.shuffle", lambda seq, _r=rng: _r.shuffle(list(seq))
            ) as _shuffle_mock:
                mutated = gene.mutate(ga_config, mutation_rate=0.5)
            if mutated.tpsl_gene is not None:
                methods_seen.add(mutated.tpsl_gene.method)

        # 50回の変異で複数の方式が現れる（スイッチが実効的に働いている証拠）
        assert len(methods_seen) >= 2

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
        high_rate = mutated_high.metadata["adaptive_mutation_rate"]
        low_rate = mutated_low.metadata["adaptive_mutation_rate"]
        assert isinstance(high_rate, (int, float))
        assert isinstance(low_rate, (int, float))

        # 高分散のrate < 低分散のrate のはず
        assert high_rate < low_rate

    def test_adaptive_mutate_with_inf_fitness_no_warning(
        self, sample_strategy_gene, ga_config
    ):
        """±inf fitness（制約違反ペナルティ）でも警告を出さずに変異できる"""
        import warnings

        class MockIndividual:
            def __init__(self, fitness_values):
                self.fitness = Mock()
                self.fitness.values = fitness_values

        # 全個体が -inf / +inf fitness（ペナルティ）の population
        all_inf_pop = [
            MockIndividual((-float("inf"),)),
            MockIndividual((float("inf"),)),
        ]
        mixed_pop = [
            MockIndividual((0.8,)),
            MockIndividual((-float("inf"),)),
            MockIndividual((1.2,)),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            mutated_all_inf = sample_strategy_gene.adaptive_mutate(
                all_inf_pop, ga_config, base_mutation_rate=0.1
            )
            mutated_mixed = sample_strategy_gene.adaptive_mutate(
                mixed_pop, ga_config, base_mutation_rate=0.1
            )

        assert isinstance(mutated_all_inf, StrategyGene)
        assert isinstance(mutated_mixed, StrategyGene)
        # 非有限値のみの場合はベースの変異率にフォールバック
        assert mutated_all_inf.metadata["adaptive_mutation_rate"] == 0.1

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

        with (
            patch("random.random", return_value=0.0),
            patch("random.randint", return_value=0),
            patch("random.choice", return_value="!="),
        ):
            mutate_conditions(gene, 1.0, ga_config)

        assert gene.long_exit_conditions[0].operator == "!="
        assert gene.short_exit_conditions[0].operator == "!="

    def test_mutate_conditions_threshold_mutation(self, ga_config):
        """数値閾値の突然変異が正しく行われることを確認"""
        gene = StrategyGene(
            indicators=[IndicatorGene(type="RSI", parameters={"period": 14})],
            long_entry_conditions=[
                Condition(left_operand="RSI", operator="<", right_operand=30.0)
            ],
        )
        ga_config.mutation_config.condition_threshold_mutation_probability = 1.0
        ga_config.mutation_config.condition_operator_switch_probability = 0.0

        # 変異実行（複数回実行して閾値が30.0から変化することを確認）
        mutated_thresholds = set()
        for _ in range(20):
            test_gene = gene.clone()
            mutate_conditions(test_gene, 1.0, ga_config)
            cond = test_gene.long_entry_conditions[0]
            if isinstance(cond, Condition) and isinstance(
                cond.right_operand, (int, float)
            ):
                mutated_thresholds.add(round(float(cond.right_operand), 4))

        # 30.0 以外の閾値が生成されていること
        assert len(mutated_thresholds) > 1

    def test_mutate_conditions_operand_swap(self, ga_config):
        """指標オペランドのスワップ変異が正しく行われることを確認"""
        gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 10}),
                IndicatorGene(type="EMA", parameters={"period": 20}),
            ],
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="SMA")
            ],
        )
        ga_config.mutation_config.condition_operand_swap_probability = 1.0
        ga_config.mutation_config.condition_threshold_mutation_probability = 0.0

        swapped = False
        for _ in range(30):
            test_gene = gene.clone()
            mutate_conditions(test_gene, 1.0, ga_config)
            cond = test_gene.long_entry_conditions[0]
            if isinstance(cond, Condition) and cond.right_operand in (
                "EMA",
                "open",
                "high",
                "low",
            ):
                swapped = True
                break

        assert swapped, "Operand swap mutation should swap operand to valid alternative"

    def test_mutate_conditions_add_and_delete(self, ga_config):
        """条件の追加と削除（枝刈り）が正しく行われることを確認"""
        gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 10}),
                IndicatorGene(type="RSI", parameters={"period": 14}),
            ],
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="SMA")
            ],
        )
        ga_config.mutation_config.condition_add_delete_probability = 1.0

        added = False
        deleted = False

        # 条件追加の検証
        for _ in range(30):
            test_gene = gene.clone()
            mutate_conditions(test_gene, 1.0, ga_config)
            if len(test_gene.long_entry_conditions) > 1:
                added = True
                break

        assert added, "Condition addition mutation should add a new valid condition"

        # 条件削除（枝刈り）の検証
        multi_cond_gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 10}),
                IndicatorGene(type="RSI", parameters={"period": 14}),
            ],
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="SMA"),
                Condition(left_operand="RSI", operator="<", right_operand=30.0),
            ],
        )
        for _ in range(30):
            test_gene = multi_cond_gene.clone()
            mutate_conditions(test_gene, 1.0, ga_config)
            if len(test_gene.long_entry_conditions) == 1:
                deleted = True
                break

        assert deleted, (
            "Condition deletion mutation should prune a condition when multiple exist"
        )

    def test_mutate_indicators_smart_condition_injection(self, ga_config):
        """新しい指標が追加された際に、その指標を利用する条件が注入されることを確認"""
        from app.services.auto_strategy.genes.operators.mutation import (
            mutate_indicators,
        )
        from app.services.auto_strategy.utils.indicator_references import (
            build_indicator_reference_name,
        )

        gene = StrategyGene(
            indicators=[IndicatorGene(type="SMA", parameters={"period": 10})],
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="SMA")
            ],
        )
        ga_config.mutation_config.indicator_add_delete_probability = 1.0
        ga_config.mutation_config.indicator_add_vs_delete_probability = 1.0
        ga_config.mutation_config.indicator_condition_injection_probability = 1.0

        injected = False
        for _ in range(30):
            test_gene = gene.clone()
            mutate_indicators(test_gene, 1.0, ga_config)
            if len(test_gene.indicators) > 1:
                new_ind = test_gene.indicators[-1]
                # 新しい指標が条件のどこかで参照されているか
                # 注入時は build_indicator_reference_name と同一の参照名が使われるため、
                # 期待値も同じ関数で構築する（id/timeframe を含む場合を考慮）
                valid_names = {build_indicator_reference_name(new_ind)}
                for cond in (
                    test_gene.long_entry_conditions + test_gene.short_entry_conditions
                ):
                    if isinstance(cond, Condition) and (
                        cond.left_operand in valid_names
                        or cond.right_operand in valid_names
                    ):
                        injected = True
                        break
            if injected:
                break

        assert injected, (
            "Smart condition injection should inject conditions using the new indicator"
        )

    def test_mutate_tool_genes_add_and_toggle(self, ga_config):
        """ToolGene の追加・パラメータ変異が動作することを確認"""
        from app.services.auto_strategy.genes.operators.mutation import (
            mutate_strategy_gene,
        )
        from app.services.auto_strategy.genes.tool import ToolGene

        gene = StrategyGene(
            tool_genes=[ToolGene(tool_name="weekend_filter", enabled=True, params={})],
        )
        ga_config.mutation_config.tool_gene_add_delete_probability = 1.0

        # 変異実行
        mutated = mutate_strategy_gene(gene, ga_config, mutation_rate=1.0)
        assert isinstance(mutated, StrategyGene)

    def test_mutate_conditions_respects_max_conditions(self, ga_config):
        """条件変異およびスマート注入が max_conditions を超過しないことを確認"""
        from app.services.auto_strategy.genes.operators.mutation import (
            mutate_conditions,
            mutate_indicators,
        )

        ga_config.max_conditions = 2
        ga_config.mutation_config.condition_add_delete_probability = 1.0
        ga_config.mutation_config.indicator_condition_injection_probability = 1.0
        ga_config.mutation_config.indicator_add_delete_probability = 1.0
        ga_config.mutation_config.indicator_add_vs_delete_probability = 1.0

        gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 10}),
                IndicatorGene(type="EMA", parameters={"period": 20}),
            ],
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="SMA"),
                Condition(left_operand="close", operator=">", right_operand="EMA"),
            ],
        )

        # すでに max_conditions (2) に達している状態で変異を実行
        for _ in range(20):
            test_gene = gene.clone()
            mutate_conditions(test_gene, 1.0, ga_config)
            assert len(test_gene.long_entry_conditions) <= ga_config.max_conditions

            test_gene2 = gene.clone()
            mutate_indicators(test_gene2, 1.0, ga_config)
            assert len(test_gene2.long_entry_conditions) <= ga_config.max_conditions

    def test_mutate_stateful_conditions_mutates_existing(self, ga_config):
        """既存 stateful 条件の lookback/cooldown/direction/enabled が変異することを確認"""
        from app.services.auto_strategy.genes.conditions import StatefulCondition
        from app.services.auto_strategy.genes.operators.mutation import (
            mutate_stateful_conditions,
        )

        gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 10}),
                IndicatorGene(type="RSI", parameters={"period": 14}),
            ],
            stateful_conditions=[
                StatefulCondition(
                    trigger_condition=Condition(
                        left_operand="close", operator=">", right_operand="SMA"
                    ),
                    follow_condition=Condition(
                        left_operand="RSI", operator="<", right_operand=30.0
                    ),
                    lookback_bars=5,
                    cooldown_bars=2,
                    direction="long",
                    enabled=True,
                )
            ],
        )
        ga_config.mutation_config.condition_operand_swap_probability = 0.0

        changed = False
        for seed in range(30):
            random.seed(seed)
            test_gene = gene.clone()
            mutate_stateful_conditions(test_gene, 1.0, ga_config)
            sc = test_gene.stateful_conditions[0]
            if (
                sc.lookback_bars != 5
                or sc.cooldown_bars != 2
                or sc.direction != "long"
                or sc.enabled is not True
                or sc.trigger_condition.operator != ">"
            ):
                changed = True
                break

        assert changed, "Stateful condition mutation should alter at least one field"

    def test_mutate_stateful_conditions_topology_add_and_delete(self, ga_config):
        """stateful 条件の追加・削除トポロジー変異が動作することを確認"""
        from app.services.auto_strategy.genes.conditions import StatefulCondition
        from app.services.auto_strategy.genes.operators.mutation import (
            mutate_stateful_conditions,
        )

        gene = StrategyGene(
            indicators=[IndicatorGene(type="SMA", parameters={"period": 10})],
            stateful_conditions=[],
        )

        added = False
        for seed in range(50):
            random.seed(seed)
            test_gene = gene.clone()
            mutate_stateful_conditions(test_gene, 1.0, ga_config)
            if len(test_gene.stateful_conditions) > 0:
                added = True
                break

        assert added, "Stateful topology mutation should add a new condition"

        multi_gene = StrategyGene(
            indicators=[IndicatorGene(type="SMA", parameters={"period": 10})],
            stateful_conditions=[
                StatefulCondition(
                    trigger_condition=Condition(
                        left_operand="close", operator=">", right_operand="SMA"
                    ),
                    follow_condition=Condition(
                        left_operand="close", operator="<", right_operand="SMA"
                    ),
                ),
                StatefulCondition(
                    trigger_condition=Condition(
                        left_operand="close", operator="<", right_operand="SMA"
                    ),
                    follow_condition=Condition(
                        left_operand="close", operator=">", right_operand="SMA"
                    ),
                ),
            ],
        )

        deleted = False
        for seed in range(100):
            random.seed(1000 + seed)
            test_gene = multi_gene.clone()
            mutate_stateful_conditions(test_gene, 1.0, ga_config)
            if len(test_gene.stateful_conditions) < 2:
                deleted = True
                break

        assert deleted, "Stateful topology mutation should delete an existing condition"

    def test_mutate_indicators_type_switch(self, ga_config):
        """指標タイプのスイッチ変異が動作することを確認"""
        from app.services.auto_strategy.genes.operators.mutation import (
            mutate_indicators,
        )

        gene = StrategyGene(
            indicators=[IndicatorGene(type="SMA", parameters={"period": 10})],
        )
        # 追加/削除を無効化してタイプスイッチのみを観察
        ga_config.mutation_config.indicator_add_delete_probability = 0.0

        switched = False
        for seed in range(100):
            random.seed(seed)
            test_gene = gene.clone()
            mutate_indicators(test_gene, 1.0, ga_config)
            if test_gene.indicators[0].type != "SMA":
                switched = True
                break

        assert switched, "Indicator type mutation should switch to another type"

    def test_mutate_indicators_timeframe_toggle(self, ga_config):
        """タイムフレーム変異（MTF有効時）が動作することを確認"""
        from app.services.auto_strategy.genes.operators.mutation import (
            mutate_indicators,
        )

        ga_config.enable_multi_timeframe = True
        ga_config.mutation_config.indicator_add_delete_probability = 0.0

        gene = StrategyGene(
            indicators=[IndicatorGene(type="SMA", parameters={"period": 10})],
        )

        toggled = False
        for seed in range(200):
            random.seed(seed)
            test_gene = gene.clone()
            mutate_indicators(test_gene, 1.0, ga_config)
            if test_gene.indicators[0].timeframe is not None:
                toggled = True
                break

        assert toggled, "Timeframe mutation should assign a timeframe when MTF enabled"

    def test_mutate_indicators_timeframe_none_when_mtf_disabled(self, ga_config):
        """MTF無効時は timeframe が None に正規化されることを確認"""
        from app.services.auto_strategy.genes.operators.mutation import (
            mutate_indicators,
        )

        ga_config.enable_multi_timeframe = False
        ga_config.mutation_config.indicator_add_delete_probability = 0.0

        gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 10}, timeframe="4h")
            ],
        )

        normalized = False
        for seed in range(200):
            random.seed(seed)
            test_gene = gene.clone()
            mutate_indicators(test_gene, 1.0, ga_config)
            if test_gene.indicators[0].timeframe is None:
                normalized = True
                break

        assert normalized, (
            "Timeframe mutation should normalize timeframe to None when MTF disabled"
        )

    def test_mutate_indicators_enabled_toggle(self, ga_config):
        """指標 enabled フラグのトグル変異が動作することを確認"""
        from app.services.auto_strategy.genes.operators.mutation import (
            mutate_indicators,
        )

        ga_config.mutation_config.indicator_add_delete_probability = 0.0

        gene = StrategyGene(
            indicators=[IndicatorGene(type="SMA", parameters={"period": 10})],
        )

        toggled = False
        for seed in range(500):
            random.seed(seed)
            test_gene = gene.clone()
            mutate_indicators(test_gene, 1.0, ga_config)
            if test_gene.indicators[0].enabled is False:
                toggled = True
                break

        assert toggled, "Enabled flag mutation should toggle the indicator off"

    def test_mutate_tool_genes_deletion(self, ga_config):
        """ToolGene の削除変異が動作し weekend_filter は保護されることを確認"""
        from app.services.auto_strategy.genes.operators.mutation import (
            mutate_strategy_gene,
        )
        from app.services.auto_strategy.genes.tool import ToolGene

        ga_config.mutation_config.tool_gene_add_delete_probability = 1.0

        gene = StrategyGene(
            tool_genes=[
                ToolGene(tool_name="weekend_filter", enabled=True, params={}),
                ToolGene(tool_name="trend_filter", enabled=True, params={}),
            ],
        )

        deleted = False
        for seed in range(50):
            random.seed(seed)
            test_gene = gene.clone()
            mutated = mutate_strategy_gene(test_gene, ga_config, mutation_rate=1.0)
            names = [t.tool_name for t in mutated.tool_genes]
            if "trend_filter" not in names:
                deleted = True
            # weekend_filter は常に存在する
            assert "weekend_filter" in names

        assert deleted, "Tool deletion mutation should remove a non-weekend tool"

    def test_mutate_condition_topology_generates_group(self, ga_config):
        """トポロジー変異で ConditionGroup が生成されることを確認"""
        from app.services.auto_strategy.genes.conditions import ConditionGroup
        from app.services.auto_strategy.genes.operators.mutation import (
            _mutate_condition_topology,
        )

        ga_config.max_conditions = 5
        ga_config.min_conditions = 1

        gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 10}),
                IndicatorGene(type="RSI", parameters={"period": 14}),
            ],
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="SMA")
            ],
        )

        group_created = False
        for seed in range(100):
            random.seed(seed)
            test_gene = gene.clone()
            _mutate_condition_topology(
                test_gene.long_entry_conditions, test_gene, ga_config, side="long"
            )
            if any(
                isinstance(c, ConditionGroup) for c in test_gene.long_entry_conditions
            ):
                group_created = True
                break

        assert group_created, "Topology mutation should generate a ConditionGroup"

    def test_mutate_entry_gene_toggles_enabled_and_priority(self):
        """EntryGene の enabled/priority 変異が動作することを確認"""
        import random as _random

        from app.services.auto_strategy.genes.entry import EntryGene

        gene = EntryGene(enabled=True, priority=1.0)

        enabled_changed = False
        priority_changed = False
        for seed in range(300):
            _random.seed(seed)
            mutated = gene.mutate(mutation_rate=1.0)
            if mutated.enabled is not True:
                enabled_changed = True
            if mutated.priority != 1.0:
                priority_changed = True
            if enabled_changed and priority_changed:
                break

        assert enabled_changed, "EntryGene mutation should toggle enabled"
        assert priority_changed, "EntryGene mutation should perturb priority"

    def test_mutate_exit_gene_toggles_enabled_and_priority(self):
        """ExitGene の enabled/priority 変異が動作することを確認"""
        import random as _random

        from app.services.auto_strategy.genes.exit import ExitGene

        gene = ExitGene(enabled=True, priority=1.0)

        enabled_changed = False
        priority_changed = False
        for seed in range(300):
            _random.seed(seed)
            mutated = gene.mutate(mutation_rate=1.0)
            if mutated.enabled is not True:
                enabled_changed = True
            if mutated.priority != 1.0:
                priority_changed = True
            if enabled_changed and priority_changed:
                break

        assert enabled_changed, "ExitGene mutation should toggle enabled"
        assert priority_changed, "ExitGene mutation should perturb priority"
