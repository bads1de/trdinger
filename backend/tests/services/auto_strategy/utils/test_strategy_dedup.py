"""戦略構造シグネチャによる重複排除のテスト"""

from __future__ import annotations

import copy

from app.services.auto_strategy.genes import (
    Condition,
    ConditionGroup,
    IndicatorGene,
    StrategyGene,
)
from app.services.auto_strategy.utils.strategy_dedup import (
    deduplicate_strategies,
    strategy_structure_signature,
)


def _make_gene(
    gene_id: str = "g1",
    *,
    indicator_period: int = 20,
    rsi_threshold: int | float = 30,
) -> StrategyGene:
    """構造をパラメータ化した戦略遺伝子を生成する"""
    return StrategyGene(
        id=gene_id,
        indicators=[
            IndicatorGene(type="SMA", parameters={"period": indicator_period}),
            IndicatorGene(type="RSI", parameters={"period": 14}),
        ],
        long_entry_conditions=[
            Condition(
                left_operand="close",
                operator=">",
                right_operand="SMA_ref",
            ),
            Condition(
                left_operand="RSI_ref",
                operator="<",
                right_operand=rsi_threshold,
            ),
        ],
        short_entry_conditions=[],
        risk_management={"position_size": 0.1},
    )


class TestStrategyStructureSignature:
    def test_identical_structures_with_different_ids_share_signature(self):
        gene_a = _make_gene("alpha")
        gene_b = _make_gene("beta")

        assert strategy_structure_signature(gene_a) == strategy_structure_signature(
            gene_b
        )

    def test_different_indicator_parameters_produce_distinct_signatures(self):
        gene_a = _make_gene(indicator_period=20)
        gene_b = _make_gene(indicator_period=25)

        assert strategy_structure_signature(gene_a) != strategy_structure_signature(
            gene_b
        )

    def test_different_threshold_types_produce_distinct_signatures(self):
        # 数値 30 と文字列 "30" は型タグで区別される
        gene_a = _make_gene(rsi_threshold=30)
        gene_b = _make_gene(rsi_threshold="30")

        assert strategy_structure_signature(gene_a) != strategy_structure_signature(
            gene_b
        )

    def test_condition_group_order_does_not_change_signature(self):
        gene_a = _make_gene()
        gene_b = _make_gene()

        cond_1 = Condition(left_operand="close", operator=">", right_operand="a")
        cond_2 = Condition(left_operand="open", operator="<", right_operand="b")
        gene_a.long_entry_conditions = [
            ConditionGroup(operator="AND", conditions=[cond_1, cond_2])
        ]
        gene_b.long_entry_conditions = [
            ConditionGroup(operator="AND", conditions=[cond_2, cond_1])
        ]

        assert strategy_structure_signature(gene_a) == strategy_structure_signature(
            gene_b
        )

    def test_different_group_operator_produces_distinct_signatures(self):
        gene_a = _make_gene()
        gene_b = _make_gene()

        cond_1 = Condition(left_operand="close", operator=">", right_operand="a")
        cond_2 = Condition(left_operand="open", operator="<", right_operand="b")
        gene_a.long_entry_conditions = [
            ConditionGroup(operator="AND", conditions=[cond_1, cond_2])
        ]
        gene_b.long_entry_conditions = [
            ConditionGroup(operator="OR", conditions=[cond_1, cond_2])
        ]

        assert strategy_structure_signature(gene_a) != strategy_structure_signature(
            gene_b
        )

    def test_timeframe_difference_produces_distinct_signatures(self):
        gene_a = _make_gene()
        gene_b = _make_gene()
        gene_b.indicators[0].timeframe = "1d"

        assert strategy_structure_signature(gene_a) != strategy_structure_signature(
            gene_b
        )

    def test_metadata_is_ignored(self):
        gene_a = _make_gene()
        gene_b = _make_gene()
        gene_b.metadata = {"created_at": "2026-01-01", "parents": ["x", "y"]}

        assert strategy_structure_signature(gene_a) == strategy_structure_signature(
            gene_b
        )

    def test_different_tpsl_produce_distinct_signatures(self):
        """TP/SL違いの戦略が同一構造と誤判定されないこと（slots dataclass 対応）"""
        from app.services.auto_strategy.genes import TPSLGene

        gene_a = _make_gene()
        gene_a.tpsl_gene = TPSLGene(take_profit_pct=0.01, stop_loss_pct=0.005)
        gene_b = _make_gene()
        gene_b.tpsl_gene = TPSLGene(take_profit_pct=0.10, stop_loss_pct=0.05)

        assert strategy_structure_signature(gene_a) != strategy_structure_signature(
            gene_b
        )

    def test_different_position_sizing_produce_distinct_signatures(self):
        """ポジションサイジング違いの戦略が同一構造と誤判定されないこと"""
        from app.services.auto_strategy.genes import PositionSizingGene

        gene_a = _make_gene()
        gene_a.position_sizing_gene = PositionSizingGene(
            method="FIXED_QUANTITY", fixed_quantity=1000
        )
        gene_b = _make_gene()
        gene_b.position_sizing_gene = PositionSizingGene(
            method="FIXED_QUANTITY", fixed_quantity=2000
        )

        assert strategy_structure_signature(gene_a) != strategy_structure_signature(
            gene_b
        )


class TestDeduplicateStrategies:
    def test_clones_are_collapsed_to_best_score(self):
        clone_low = _make_gene("c1")
        clone_high = _make_gene("c2")
        other = _make_gene("o1")
        other.indicators[0].parameters["period"] = 50
        strategies = [clone_low, other, clone_high]
        scores = [0.5, 0.7, 0.9]

        deduplicated = deduplicate_strategies(strategies, scores)

        # シグネチャごとに1件、最高スコアの個体が残る
        assert len(deduplicated) == 2
        assert deduplicated[0] == (clone_high, 0.9)
        assert deduplicated[1] == (other, 0.7)

    def test_exclude_drops_structural_duplicates_of_excluded_strategy(self):
        best = _make_gene("best")
        clone = _make_gene("clone")
        other = _make_gene("other")
        other.indicators[0].parameters["period"] = 50

        deduplicated = deduplicate_strategies(
            [best, clone, other],
            [1.0, 0.9, 0.8],
            exclude=[best],
        )

        assert [gene.id for gene, _ in deduplicated] == ["other"]

    def test_first_seen_order_is_preserved(self):
        first = _make_gene("first")
        second = _make_gene("second")
        second.indicators[0].parameters["period"] = 50
        clone_of_second = _make_gene("clone2")
        clone_of_second.indicators[0].parameters["period"] = 50

        deduplicated = deduplicate_strategies(
            [first, second, clone_of_second],
            [0.3, 0.5, 0.9],
        )

        assert [gene.id for gene, _ in deduplicated] == ["first", "clone2"]
        assert deduplicated[1][1] == 0.9

    def test_max_count_limits_output_after_dedup(self):
        genes = []
        scores = []
        for i in range(10):
            gene = _make_gene(f"g{i}")
            gene.indicators[0].parameters["period"] = 10 + i
            genes.append(gene)
            scores.append(0.1 * i)

        deduplicated = deduplicate_strategies(genes, scores, max_count=3)

        assert len(deduplicated) == 3
        assert [gene.id for gene, _ in deduplicated] == ["g0", "g1", "g2"]

    def test_missing_scores_fall_back_to_zero(self):
        gene_a = _make_gene("a")
        gene_b = _make_gene("b")
        gene_b.indicators[0].parameters["period"] = 50

        deduplicated = deduplicate_strategies([gene_a, gene_b], [])

        assert deduplicated == [(gene_a, 0.0), (gene_b, 0.0)]

    def test_deep_copies_are_recognized_as_duplicates(self):
        original = _make_gene()
        clone = copy.deepcopy(original)
        clone.id = "different-id"

        deduplicated = deduplicate_strategies([original, clone], [0.4, 0.6])

        assert len(deduplicated) == 1
        assert deduplicated[0] == (clone, 0.6)
