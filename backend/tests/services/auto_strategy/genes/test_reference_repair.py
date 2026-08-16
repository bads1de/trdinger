"""
条件参照整合性（reference repair）のテスト

交叉・突然変異後に、自身の指標リストへ参照切れ（dangling reference）を
含む条件が除去されることを検証します。
"""

import pytest

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.genes import (
    Condition,
    ConditionGroup,
    IndicatorGene,
    StrategyGene,
)
from app.services.auto_strategy.genes.conditions import StatefulCondition
from app.services.auto_strategy.genes.operators.crossover import (
    single_point_crossover,
    uniform_crossover,
)
from app.services.auto_strategy.genes.strategy import (
    _is_resolvable_operand,
    _operand_reference_name,
)

_EMA_ID = "aaaa1111-2222-3333-4444-555566667777"
_RSI_ID = "bbbb2222-3333-4444-5555-666677778888"


def _make_gene(
    indicators: list[IndicatorGene],
    long_entry: list | None = None,
    long_exit: list | None = None,
    stateful: list | None = None,
) -> StrategyGene:
    return StrategyGene(
        id="gene-test",
        indicators=indicators,
        long_entry_conditions=long_entry or [],
        long_exit_conditions=long_exit or [],
        stateful_conditions=stateful or [],
        risk_management={"position_size": 0.1},
        metadata={},
    )


def _dangling_operands(gene: StrategyGene) -> set[str]:
    """遺伝子内の条件を走査し、解決不能なオペランド名を返す。"""
    valid = gene.valid_operand_names()
    dangling: set[str] = set()

    def scan(item: object) -> None:
        if isinstance(item, ConditionGroup):
            for sub in item.conditions:
                scan(sub)
            return
        for operand in (item.left_operand, item.right_operand):  # type: ignore[attr-defined]
            name = _operand_reference_name(operand)
            if name is not None and not _is_resolvable_operand(name, valid):
                dangling.add(name)

    for conditions in (
        gene.long_entry_conditions,
        gene.short_entry_conditions,
        gene.long_exit_conditions,
        gene.short_exit_conditions,
    ):
        for item in conditions:
            scan(item)
    return dangling


class TestRepairConditionReferences:
    """repair_condition_references のテスト"""

    def test_repair_removes_dangling_single_condition(self):
        """自分の指標に存在しない参照を含む条件のみ除去される"""
        gene = _make_gene(
            indicators=[
                IndicatorGene(type="EMA", parameters={"period": 20}, id=_EMA_ID)
            ],
            long_entry=[
                Condition(
                    left_operand="close", operator=">", right_operand="SMA_ffffffff"
                ),
                Condition(
                    left_operand="close", operator=">", right_operand="EMA_aaaa1111"
                ),
            ],
        )

        removed = gene.repair_condition_references()

        assert removed == 1
        assert len(gene.long_entry_conditions) == 1
        kept = gene.long_entry_conditions[0]
        assert kept.right_operand == "EMA_aaaa1111"

    def test_repair_keeps_all_reference_variants(self):
        """裸名・ID付き・出力インデックス付き参照はすべて保持される"""
        gene = _make_gene(
            indicators=[
                IndicatorGene(type="EMA", parameters={"period": 20}, id=_EMA_ID)
            ],
            long_entry=[
                Condition(left_operand="EMA", operator=">", right_operand="close"),
                Condition(
                    left_operand="EMA_aaaa1111", operator=">", right_operand="close"
                ),
                Condition(
                    left_operand="EMA_aaaa1111_0", operator=">", right_operand=50
                ),
                Condition(left_operand="close", operator=">", right_operand="0.5"),
            ],
        )

        removed = gene.repair_condition_references()

        assert removed == 0
        assert len(gene.long_entry_conditions) == 4

    def test_repair_filters_dangling_members_inside_group(self):
        """グループ内の参照切れ条件のみが除去され、グループ自体は保持される"""
        gene = _make_gene(
            indicators=[
                IndicatorGene(type="EMA", parameters={"period": 20}, id=_EMA_ID)
            ],
            long_entry=[
                ConditionGroup(
                    operator="OR",
                    conditions=[
                        Condition(
                            left_operand="close",
                            operator=">",
                            right_operand="EMA_aaaa1111",
                        ),
                        Condition(
                            left_operand="SMA_ffffffff",
                            operator=">",
                            right_operand="close",
                        ),
                    ],
                )
            ],
        )

        removed = gene.repair_condition_references()

        assert removed == 1
        group = gene.long_entry_conditions[0]
        assert isinstance(group, ConditionGroup)
        assert len(group.conditions) == 1

    def test_repair_removes_all_dangling_group(self):
        """全メンバーが参照切れのグループは丸ごと除去される"""
        gene = _make_gene(
            indicators=[
                IndicatorGene(type="EMA", parameters={"period": 20}, id=_EMA_ID)
            ],
            long_entry=[
                ConditionGroup(
                    operator="AND",
                    conditions=[
                        Condition(
                            left_operand="SMA_ffffffff",
                            operator=">",
                            right_operand="close",
                        ),
                        Condition(
                            left_operand="RSI_00000000", operator="<", right_operand=30
                        ),
                    ],
                )
            ],
        )

        removed = gene.repair_condition_references()

        # 葉条件2つとグループ自体の除去で3カウント
        assert removed == 3
        assert gene.long_entry_conditions == []

    def test_repair_removes_dangling_stateful_condition(self):
        """トリガー条件が参照切れのステートフル条件は除去される"""
        gene = _make_gene(
            indicators=[
                IndicatorGene(type="EMA", parameters={"period": 20}, id=_EMA_ID)
            ],
            stateful=[
                StatefulCondition(
                    trigger_condition=Condition(
                        left_operand="close", operator=">", right_operand="SMA_ffffffff"
                    ),
                    follow_condition=Condition(
                        left_operand="close", operator=">", right_operand="EMA_aaaa1111"
                    ),
                )
            ],
        )

        removed = gene.repair_condition_references()

        assert removed == 1
        assert gene.stateful_conditions == []

    def test_repair_after_indicator_removal(self):
        """突然変異による指標削除後に残った条件参照が除去される"""
        gene = _make_gene(
            indicators=[
                IndicatorGene(type="EMA", parameters={"period": 20}, id=_EMA_ID),
                IndicatorGene(type="RSI", parameters={"period": 14}, id=_RSI_ID),
            ],
            long_entry=[
                Condition(
                    left_operand="close", operator=">", right_operand="EMA_aaaa1111"
                ),
                Condition(left_operand="RSI_bbbb2222", operator="<", right_operand=30),
            ],
        )

        gene.indicators.pop(1)
        removed = gene.repair_condition_references()

        assert removed == 1
        assert len(gene.long_entry_conditions) == 1
        assert gene.long_entry_conditions[0].right_operand == "EMA_aaaa1111"


class TestOperatorIntegrity:
    """交叉・突然変異後の参照整合性のテスト"""

    @pytest.fixture
    def parent1(self) -> StrategyGene:
        return _make_gene(
            indicators=[
                IndicatorGene(type="EMA", parameters={"period": 20}, id=_EMA_ID)
            ],
            long_entry=[
                Condition(
                    left_operand="close", operator=">", right_operand="EMA_aaaa1111"
                )
            ],
            long_exit=[
                Condition(
                    left_operand="close", operator="<", right_operand="EMA_aaaa1111"
                )
            ],
        )

    @pytest.fixture
    def parent2(self) -> StrategyGene:
        return _make_gene(
            indicators=[
                IndicatorGene(type="RSI", parameters={"period": 14}, id=_RSI_ID)
            ],
            long_entry=[
                Condition(left_operand="RSI_bbbb2222", operator="<", right_operand=30)
            ],
            long_exit=[
                Condition(left_operand="RSI_bbbb2222", operator=">", right_operand=70)
            ],
        )

    def test_uniform_crossover_children_have_no_dangling(self, parent1, parent2):
        """ユニフォーム交叉の子は参照切れ条件を持たない"""
        for _ in range(30):
            child1, child2 = uniform_crossover(
                StrategyGene, parent1, parent2, GAConfig()
            )
            assert _dangling_operands(child1) == set()
            assert _dangling_operands(child2) == set()

    def test_single_point_crossover_children_have_no_dangling(self, parent1, parent2):
        """一点交叉の子は参照切れ条件を持たない"""
        for _ in range(30):
            child1, child2 = single_point_crossover(
                StrategyGene, parent1, parent2, GAConfig()
            )
            assert _dangling_operands(child1) == set()
            assert _dangling_operands(child2) == set()

    def test_mutation_keeps_references(self, parent1):
        """突然変異後も参照切れ条件が残らない"""
        for _ in range(30):
            mutated = parent1.mutate(GAConfig(), mutation_rate=1.0)
            assert _dangling_operands(mutated) == set()
