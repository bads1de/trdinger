"""
条件オペランドのスケール互換性（scale compat / repair）のテスト

価格オペランド（close 等）と値域が桁違いの指標（FUNDING_RATE_LEVEL 等）の
直接比較が恒真/恒偽の退化条件になるため、生成器では比較相手を制限し、
演算子適用後は repair_condition_scales で書き直すことを検証します。
"""

import pytest

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.generators.condition_generator import (
    ConditionGenerator,
)
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
from app.services.auto_strategy.genes.strategy import _operand_reference_name
from app.services.auto_strategy.utils.indicator_references import (
    build_indicator_reference_name,
)
from app.services.auto_strategy.utils.scale_compat import (
    default_threshold_for,
    is_close_comparable_type,
    is_price_operand,
)

_EMA_ID = "aaaa1111-2222-3333-4444-555566667777"
_FR_ID = "cccc3333-4444-5555-6666-777788889999"


def _make_gene(
    indicators: list[IndicatorGene],
    long_entry: list | None = None,
    short_entry: list | None = None,
    stateful: list | None = None,
) -> StrategyGene:
    return StrategyGene(
        id="gene-test",
        indicators=indicators,
        long_entry_conditions=long_entry or [],
        short_entry_conditions=short_entry or [],
        stateful_conditions=stateful or [],
        risk_management={"position_size": 0.1},
        metadata={},
    )


def _scale_violations(gene: StrategyGene) -> set[str]:
    """遺伝子内の条件を走査し、価格×非価格スケール比較を返す。"""
    # 修復ロジックと同じ規約でオペランド名を指標型へ解決する
    name_to_type: dict[str, str] = {}
    for indicator in gene.indicators:
        name_to_type.setdefault(indicator.type, indicator.type)
        timeframe = getattr(indicator, "timeframe", None)
        if timeframe:
            name_to_type.setdefault(f"{indicator.type}_{timeframe}", indicator.type)
        name_to_type[build_indicator_reference_name(indicator)] = indicator.type

    def resolve_type(name: str) -> str | None:
        resolved = name_to_type.get(name)
        if resolved is None:
            stem, _, suffix = name.rpartition("_")
            if stem and suffix.isdigit():
                resolved = name_to_type.get(stem)
        return resolved

    def leaf_violation(cond: Condition) -> str | None:
        for price_attr, other_attr in (
            ("left_operand", "right_operand"),
            ("right_operand", "left_operand"),
        ):
            if not is_price_operand(getattr(cond, price_attr)):
                continue
            other_name = _operand_reference_name(getattr(cond, other_attr))
            if other_name is None:
                continue
            try:
                float(other_name)
                continue  # 数値リテラル
            except ValueError:
                pass
            other_type = resolve_type(other_name)
            if other_type is None:
                continue  # 自身の指標以外（参照修復の管轄）
            if not is_close_comparable_type(other_type):
                return f"{getattr(cond, price_attr)}:{other_name}"
        return None

    violations: set[str] = set()

    def scan(item: object) -> None:
        if isinstance(item, ConditionGroup):
            for sub in item.conditions:
                scan(sub)
            return
        violation = leaf_violation(item)  # type: ignore[arg-type]
        if violation:
            violations.add(violation)

    for conditions in (
        gene.long_entry_conditions,
        gene.short_entry_conditions,
        gene.long_exit_conditions,
        gene.short_exit_conditions,
    ):
        for item in conditions:
            scan(item)
    for stateful in gene.stateful_conditions:
        scan(stateful.trigger_condition)
        scan(stateful.follow_condition)
    return violations


class TestScaleCompatUtil:
    """scale_compat ユーティリティのテスト"""

    @pytest.mark.parametrize(
        "indicator_type",
        [
            "SMA",
            "EMA",
            "HMA",
            "KAMA",
            "T3",
            "ZLEMA",
            "VWAP",
            "SUPERTREND",
            "BBANDS",
        ],
    )
    def test_price_overlays_are_close_comparable(self, indicator_type: str):
        """価格オーバーレイは close と比較可能"""
        assert is_close_comparable_type(indicator_type)

    @pytest.mark.parametrize(
        "indicator_type",
        [
            "FUNDING_RATE_LEVEL",
            "SQUEEZE_PROBABILITY",
            "ADX",
            "CHOP",
            "RSI",
            "ATR",
            "LSR_ROC",
            "MACD",
            "FISHER",
            "TSI",
            "PPO",
            "WHALE_DIVERGENCE",
        ],
    )
    def test_non_price_indicators_are_not_comparable(self, indicator_type: str):
        """値域が価格と桁違いの指標は close と比較できない"""
        assert not is_close_comparable_type(indicator_type)

    def test_default_threshold_prefers_registry_profile(self):
        """レジストリの normal プロファイル閾値が優先される"""
        # FUNDING_RATE_LEVEL normal: long_gt=0.0, short_lt=0.01
        assert default_threshold_for("FUNDING_RATE_LEVEL", bullish=True) == 0.0
        assert default_threshold_for("FUNDING_RATE_LEVEL", bullish=False) == 0.01

    def test_default_threshold_by_scale_type(self):
        """閾値未定義の指標はスケール型の中心値を使う"""
        # ATR は PRICE_RATIO 無閾値 → 1.0
        assert default_threshold_for("ATR", bullish=True) == 1.0
        # 未知の指標型は 0.0
        assert default_threshold_for("NO_SUCH_INDICATOR", bullish=True) == 0.0

    def test_price_operand_detection(self):
        """価格フィールドの判定（volume は含めない）"""
        assert is_price_operand("close")
        assert is_price_operand("Close")
        assert is_price_operand("open")
        assert not is_price_operand("volume")
        assert not is_price_operand("EMA_aaaa1111")
        assert not is_price_operand(50.0)


class TestGeneratorScaleGuards:
    """生成器のスケールガードのテスト"""

    def test_fallback_never_selects_non_price_indicator(self):
        """フォールバック条件の比較相手が非価格指標にならない"""
        gen = ConditionGenerator()
        indicators = [
            IndicatorGene(
                type="FUNDING_RATE_LEVEL", parameters={"period": 8}, id=_FR_ID
            ),
            IndicatorGene(type="SQUEEZE_PROBABILITY", parameters={"period": 20}),
            IndicatorGene(type="HMA", parameters={"period": 21}),
            IndicatorGene(type="SUPERTREND", parameters={"period": 10}),
        ]
        for _ in range(200):
            cond = gen._build_fallback_condition("long", indicators, purpose="entry")
            assert is_price_operand(cond.left_operand)
            other = _operand_reference_name(cond.right_operand)
            if other in ("open", "close", "high", "low"):
                continue
            base = other.split("_")[0] if other else ""
            assert is_close_comparable_type(base), other

    def test_select_trend_indicator_returns_none_without_price_scale(self):
        """価格スケール指標が無い場合はフォールバック候補を選ばない"""
        gen = ConditionGenerator()
        indicators = [
            IndicatorGene(
                type="FUNDING_RATE_LEVEL", parameters={"period": 8}, id=_FR_ID
            ),
            IndicatorGene(type="ADX", parameters={"period": 14}),
        ]
        for _ in range(50):
            assert gen._select_trend_indicator(indicators) is None

    def test_side_condition_for_atr_does_not_use_close(self):
        """ATR（PRICE_RATIO 無閾値）の閾値に close を使わない"""
        gen = ConditionGenerator()
        atr = IndicatorGene(type="ATR", parameters={"period": 14})
        conds = gen._create_side_conditions(atr, "long")
        assert conds[0].right_operand != "close"
        assert conds[0].right_operand == 1.0

    def test_side_condition_for_price_overlay_uses_close(self):
        """価格オーバーレイ（EMA）は close と比較する"""
        gen = ConditionGenerator()
        ema = IndicatorGene(type="EMA", parameters={"period": 20}, id=_EMA_ID)
        conds = gen._create_side_conditions(ema, "long")
        assert conds[0].right_operand == "close"


class TestRepairConditionScales:
    """repair_condition_scales のテスト"""

    def _fr_gene(self, long_entry: list) -> StrategyGene:
        return _make_gene(
            indicators=[
                IndicatorGene(
                    type="FUNDING_RATE_LEVEL", parameters={"period": 8}, id=_FR_ID
                )
            ],
            long_entry=long_entry,
        )

    def test_repair_rewrites_price_left_operand(self):
        """close > FUNDING_RATE_LEVEL が閾値比較へ書き直される"""
        gene = self._fr_gene(
            [
                Condition(
                    left_operand="close",
                    operator=">",
                    right_operand="FUNDING_RATE_LEVEL_cccc3333",
                )
            ]
        )

        repaired = gene.repair_condition_scales()

        assert repaired == 1
        cond = gene.long_entry_conditions[0]
        assert cond.left_operand == "FUNDING_RATE_LEVEL_cccc3333"
        assert cond.right_operand == 0.0  # normal long_gt
        assert cond.operator == ">"

    def test_repair_rewrites_price_right_operand(self):
        """FUNDING_RATE_LEVEL < close の右辺価格も書き直される"""
        gene = self._fr_gene(
            [
                Condition(
                    left_operand="FUNDING_RATE_LEVEL_cccc3333",
                    operator="<",
                    right_operand="close",
                )
            ]
        )

        repaired = gene.repair_condition_scales()

        assert repaired == 1
        cond = gene.long_entry_conditions[0]
        assert cond.left_operand == "FUNDING_RATE_LEVEL_cccc3333"
        assert cond.right_operand == 0.01  # normal short_lt
        assert cond.operator == "<"

    def test_repair_preserves_output_index_suffix(self):
        """複数出力指標の出力インデックス付き参照は接尾辞を保持する"""
        gene = _make_gene(
            indicators=[
                IndicatorGene(
                    type="MACD",
                    parameters={"fast": 12, "slow": 26},
                    id="abcd1234-0000-0000-0000-000000000000",
                )
            ],
            long_entry=[
                Condition(
                    left_operand="close",
                    operator=">",
                    right_operand="MACD_abcd1234_0",
                )
            ],
        )

        repaired = gene.repair_condition_scales()

        assert repaired == 1
        cond = gene.long_entry_conditions[0]
        assert cond.left_operand == "MACD_abcd1234_0"
        assert cond.right_operand == 0.0  # MACD は 0 中心

    def test_repair_inside_group(self):
        """グループ内の退化条件も書き直される"""
        gene = self._fr_gene(
            [
                ConditionGroup(
                    operator="AND",
                    conditions=[
                        Condition(
                            left_operand="close",
                            operator=">",
                            right_operand="EMA_aaaa1111",
                        ),
                        Condition(
                            left_operand="close",
                            operator=">",
                            right_operand="FUNDING_RATE_LEVEL_cccc3333",
                        ),
                    ],
                )
            ]
        )
        # EMA を指標として追加して参照を有効にする
        gene.indicators.append(
            IndicatorGene(type="EMA", parameters={"period": 20}, id=_EMA_ID)
        )

        repaired = gene.repair_condition_scales()

        assert repaired == 1
        group = gene.long_entry_conditions[0]
        assert isinstance(group, ConditionGroup)
        assert group.conditions[0].right_operand == "EMA_aaaa1111"
        assert group.conditions[1].left_operand == "FUNDING_RATE_LEVEL_cccc3333"

    def test_repair_stateful_conditions(self):
        """ステートフル条件のトリガー/フォローも書き直す"""
        gene = self._fr_gene([])
        gene.stateful_conditions = [
            StatefulCondition(
                trigger_condition=Condition(
                    left_operand="close",
                    operator=">",
                    right_operand="FUNDING_RATE_LEVEL_cccc3333",
                ),
                follow_condition=Condition(
                    left_operand="close", operator=">", right_operand="open"
                ),
            )
        ]

        repaired = gene.repair_condition_scales()

        assert repaired == 1
        stateful = gene.stateful_conditions[0]
        assert stateful.trigger_condition.left_operand == (
            "FUNDING_RATE_LEVEL_cccc3333"
        )
        assert stateful.trigger_condition.right_operand == 0.0
        # close vs open は価格同士なので無変更
        assert stateful.follow_condition.right_operand == "open"

    def test_repair_keeps_valid_conditions(self):
        """価格オーバーレイ比較・数値比較は書き直されない"""
        gene = _make_gene(
            indicators=[
                IndicatorGene(type="EMA", parameters={"period": 20}, id=_EMA_ID)
            ],
            long_entry=[
                Condition(
                    left_operand="close", operator=">", right_operand="EMA_aaaa1111"
                ),
                Condition(left_operand="close", operator=">", right_operand=0.5),
                Condition(left_operand="close", operator=">", right_operand="open"),
            ],
        )

        repaired = gene.repair_condition_scales()

        assert repaired == 0
        assert len(gene.long_entry_conditions) == 3

    def test_repair_is_idempotent(self):
        """2回目の呼び出しは何も書き直さない"""
        gene = self._fr_gene(
            [
                Condition(
                    left_operand="close",
                    operator=">",
                    right_operand="FUNDING_RATE_LEVEL_cccc3333",
                )
            ]
        )

        assert gene.repair_condition_scales() == 1
        assert gene.repair_condition_scales() == 0


class TestOperatorScaleIntegrity:
    """交叉・突然変異後のスケール整合性のテスト"""

    @pytest.fixture
    def legacy_parent(self) -> StrategyGene:
        """スケールガード導入前のシード相当（退化条件を含む）"""
        return _make_gene(
            indicators=[
                IndicatorGene(
                    type="FUNDING_RATE_LEVEL", parameters={"period": 8}, id=_FR_ID
                )
            ],
            long_entry=[
                Condition(
                    left_operand="close",
                    operator=">",
                    right_operand="FUNDING_RATE_LEVEL_cccc3333",
                )
            ],
            short_entry=[
                Condition(
                    left_operand="FUNDING_RATE_LEVEL_cccc3333",
                    operator="<",
                    right_operand="close",
                )
            ],
        )

    @pytest.fixture
    def clean_parent(self) -> StrategyGene:
        return _make_gene(
            indicators=[
                IndicatorGene(type="EMA", parameters={"period": 20}, id=_EMA_ID)
            ],
            long_entry=[
                Condition(
                    left_operand="close", operator=">", right_operand="EMA_aaaa1111"
                )
            ],
        )

    def test_uniform_crossover_children_have_no_scale_violation(
        self, legacy_parent, clean_parent
    ):
        """ユニフォーム交叉の子に退化条件が残らない"""
        for _ in range(30):
            child1, child2 = uniform_crossover(
                StrategyGene, legacy_parent, clean_parent, GAConfig()
            )
            assert _scale_violations(child1) == set()
            assert _scale_violations(child2) == set()

    def test_single_point_crossover_children_have_no_scale_violation(
        self, legacy_parent, clean_parent
    ):
        """一点交叉の子に退化条件が残らない"""
        for _ in range(30):
            child1, child2 = single_point_crossover(
                StrategyGene, legacy_parent, clean_parent, GAConfig()
            )
            assert _scale_violations(child1) == set()
            assert _scale_violations(child2) == set()

    def test_mutation_keeps_scale_validity(self, legacy_parent):
        """突然変異後も退化条件が残らない"""
        for _ in range(30):
            mutated = legacy_parent.mutate(GAConfig(), mutation_rate=1.0)
            assert _scale_violations(mutated) == set()
