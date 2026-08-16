"""
指標分類とMTF条件生成のテスト

_get_indicator_type のスケールベース分類（TREND バケット復活）と、
MTF 戦略の指標コピー登録・参照整合性を検証します。
"""

import pytest

from app.services.auto_strategy.config.constants import IndicatorType
from app.services.auto_strategy.config.ga_config import GAConfig
from app.services.auto_strategy.generators.condition_generator import (
    ConditionGenerator,
)
from app.services.auto_strategy.generators.mtf_strategy import MTFStrategy
from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.services.auto_strategy.genes import IndicatorGene, StrategyGene
from app.services.auto_strategy.genes.strategy import (
    _is_resolvable_operand,
    _operand_reference_name,
)
from app.services.auto_strategy.utils.indicator_references import (
    build_indicator_reference_name,
)

_EMA_ID = "aaaa1111-2222-3333-4444-555566667777"
_RSI_ID = "bbbb2222-3333-4444-5555-666677778888"
_FR_ID = "cccc3333-0000-0000-0000-000000000000"

_MTF_CFG = GAConfig(
    enable_multi_timeframe=True,
    available_timeframes=["1d"],
    max_indicators=6,
)


def _gene(
    type_: str, id_: str, parameters: dict | None = None, timeframe: str | None = None
) -> IndicatorGene:
    return IndicatorGene(
        type=type_,
        parameters=parameters or {"period": 20},
        id=id_,
        timeframe=timeframe,
        enabled=True,
    )


def _dangling_operands(gene: StrategyGene) -> set[str]:
    """遺伝子内の条件を走査し、解決不能なオペランド名を返す。"""
    valid = gene.valid_operand_names()
    dangling: set[str] = set()

    def scan(item: object) -> None:
        if hasattr(item, "conditions"):
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


class TestIndicatorClassification:
    """_get_indicator_type のスケールベース分類のテスト"""

    @pytest.mark.parametrize(
        "indicator_type",
        ["SMA", "EMA", "HMA", "KAMA", "VWAP", "SUPERTREND", "BBANDS", "T3", "ZLEMA"],
    )
    def test_price_overlays_classified_as_trend(self, indicator_type: str):
        """価格オーバーレイは TREND に分類される"""
        gen = ConditionGenerator()
        assert (
            gen._get_indicator_type(_gene(indicator_type, _EMA_ID))
            is IndicatorType.TREND
        )

    @pytest.mark.parametrize(
        "indicator_type",
        [
            "RSI",
            "MACD",
            "ADX",
            "CHOP",
            "STOCH",
            "FUNDING_RATE_LEVEL",
            "SQUEEZE_PROBABILITY",
            "LSR_ROC",
            "TREND_QUALITY",
            "ATR",
        ],
    )
    def test_oscillators_and_derivatives_classified_as_momentum(
        self, indicator_type: str
    ):
        """オシレーター・派生指標は MOMENTUM に分類される"""
        gen = ConditionGenerator()
        assert (
            gen._get_indicator_type(_gene(indicator_type, _RSI_ID))
            is IndicatorType.MOMENTUM
        )

    def test_dynamic_classify_has_non_empty_trend_bucket(self):
        """混合プールで TREND バケットが空にならない"""
        gen = ConditionGenerator()
        indicators = [
            _gene("EMA", _EMA_ID),
            _gene("RSI", _RSI_ID),
            _gene("FUNDING_RATE_LEVEL", _FR_ID),
        ]
        classified = gen._dynamic_classify(indicators)
        assert [i.type for i in classified[IndicatorType.TREND]] == ["EMA"]
        assert {i.type for i in classified[IndicatorType.MOMENTUM]} == {
            "RSI",
            "FUNDING_RATE_LEVEL",
        }


class TestTrendFollowRevival:
    """トレンドフォローパターン復活のテスト"""

    def test_trend_follow_pattern_generated(self):
        """close vs トレンド指標 AND モメンタム閾値 のパターンが生成される"""
        gen = ConditionGenerator()
        indicators = [_gene("EMA", _EMA_ID), _gene("RSI", _RSI_ID)]
        ema_ref = build_indicator_reference_name(indicators[0])

        def and_groups(item: object, acc: list) -> None:
            """条件ツリーから AND グループを再帰的に収集する"""
            if hasattr(item, "conditions"):
                if item.operator == "AND":
                    acc.append(item)
                for sub in item.conditions:
                    and_groups(sub, acc)

        found = False
        for _ in range(50):
            longs, shorts, _ = gen.generate_balanced_conditions(indicators)
            for conditions in (longs, shorts):
                groups: list = []
                for item in conditions:
                    and_groups(item, groups)
                for group in groups:
                    leaves = group.conditions
                    has_close_vs_ema = any(
                        str(getattr(leaf, "left_operand", "")).lower() == "close"
                        and str(getattr(leaf, "right_operand", "")) == ema_ref
                        for leaf in leaves
                    )
                    has_rsi_threshold = any(
                        str(getattr(leaf, "left_operand", "")).startswith("RSI")
                        and isinstance(leaf.right_operand, (int, float))
                        for leaf in leaves
                    )
                    if has_close_vs_ema and has_rsi_threshold:
                        found = True
        assert found, "トレンドフォロー（close>EMA AND RSI閾値）が一度も生成されない"

    def test_fallback_uses_price_overlay_only(self):
        """フォールバック条件の比較相手は常に価格オーバーレイか価格列"""
        gen = ConditionGenerator()
        indicators = [
            _gene("EMA", _EMA_ID),
            _gene("RSI", _RSI_ID),
            _gene("FUNDING_RATE_LEVEL", _FR_ID),
        ]
        for _ in range(100):
            cond = gen._build_fallback_condition("long", indicators)
            other = str(cond.right_operand)
            assert other in ("open", "close", "high", "low") or other.startswith(
                "EMA"
            ), f"フォールバック比較相手が不正: {other}"


class TestMTFStrategy:
    """MTF条件生成と指標コピー登録のテスト"""

    def _mtf_generator(self, max_indicators: int = 10) -> ConditionGenerator:
        config = GAConfig(
            enable_multi_timeframe=True,
            available_timeframes=["1d"],
            max_indicators=max_indicators,
        )
        gen = ConditionGenerator(ga_config=config)
        gen.set_context(timeframe="1h")
        return gen

    def test_mtf_disabled_returns_empty(self):
        """enable_multi_timeframe 無効時は何も生成しない"""
        gen = ConditionGenerator(ga_config=GAConfig())
        strategy = MTFStrategy(gen)
        longs, shorts, copies = strategy.generate_conditions(
            [_gene("EMA", _EMA_ID), _gene("RSI", _RSI_ID)]
        )
        assert longs == [] and shorts == [] and copies == []

    def test_mtf_generates_conditions_and_copies(self):
        """有効時は条件と上位足指標コピーを返す"""
        gen = self._mtf_generator()
        strategy = MTFStrategy(gen)
        indicators = [_gene("EMA", _EMA_ID), _gene("RSI", _RSI_ID)]
        longs, shorts, copies = strategy.generate_conditions(indicators)

        assert longs, "MTFロング条件が生成されない"
        assert copies, "上位足指標コピーが返されない"
        copy = copies[0]
        assert copy.type == "EMA"
        assert copy.timeframe in ("4h", "1d")

        # 条件の全オペランドはコピー登録後に参照整合する
        operands = MTFStrategy._collect_operand_names(longs + shorts)
        valid = {ind.type for ind in indicators + copies}
        valid |= {build_indicator_reference_name(ind) for ind in indicators + copies}
        for name in operands:
            if name.lower() in ("close", "open", "high", "low"):
                continue
            try:
                float(name)
                continue
            except ValueError:
                pass
            assert _is_resolvable_operand(name, valid), f"解決不能な参照: {name}"

    def test_mtf_respects_max_indicators(self):
        """指標数上限を超えてコピーを作らない"""
        gen = self._mtf_generator(max_indicators=3)
        strategy = MTFStrategy(gen)
        indicators = [
            _gene("EMA", _EMA_ID),
            _gene("RSI", _RSI_ID),
            _gene("SMA", "dddd4444-0000-0000-0000-000000000000"),
        ]
        _, _, copies = strategy.generate_conditions(indicators)
        assert len(indicators) + len(copies) <= 3

    def test_mtf_reuses_existing_higher_tf_indicator(self):
        """既存の上位足指標と重複するコピーを作らず、既存分を再利用する"""
        gen = self._mtf_generator()
        gen.set_context(timeframe="4h")  # 上位足は 1d に確定
        strategy = MTFStrategy(gen)
        indicators = [
            _gene("EMA", _EMA_ID),
            _gene("EMA", "eeee5555-0000-0000-0000-000000000000", timeframe="1d"),
            _gene("RSI", _RSI_ID),
        ]
        usable, new_copies = strategy._create_mtf_indicators(
            indicators, indicators, "1d"
        )
        # 既存の EMA_1d が使われ、新規コピーは作られない
        assert any(
            getattr(ind, "timeframe", None) == "1d" and ind.type == "EMA"
            for ind in usable
        )
        assert new_copies == [] or all(
            (ind.type, "1d") != ("EMA", "1d") for ind in new_copies
        )
        _, _, copies = strategy.generate_conditions(indicators)
        assert all(c.type != "EMA" for c in copies), "EMA_1d が二重登録される"


class TestGeneGenerationWithMTF:
    """遺伝子生成全体のMTF登録のテスト"""

    def _generator(self, max_indicators: int = 6) -> RandomGeneGenerator:
        config = GAConfig(
            enable_multi_timeframe=True,
            available_timeframes=["1d"],
            mtf_indicator_probability=0.5,
            max_indicators=max_indicators,
            population_size=4,
            generations=1,
        )
        generator = RandomGeneGenerator(config=config)
        generator._initialize_caches()
        return generator

    def test_generated_gene_has_no_dangling_mtf_references(self):
        """MTF有効時の生成遺伝子は上位足参照も含めて参照整合する"""
        generator = self._generator()
        mtf_genes = 0
        for _ in range(20):
            gene = generator.generate_random_gene()
            assert _dangling_operands(gene) == set()
            if any(getattr(i, "timeframe", None) for i in gene.indicators):
                mtf_genes += 1
        assert mtf_genes > 0, "20回の生成でMTF指標が一度も作られない"

    def test_gene_indicator_count_within_limit(self):
        """MTFコピー追加後も指標数上限を守る"""
        generator = self._generator(max_indicators=4)
        for _ in range(10):
            gene = generator.generate_random_gene()
            assert len(gene.indicators) <= 4

    def test_mtf_references_survive_operators(self):
        """交叉・変異後もMTF参照の整合性が保たれる"""
        from app.services.auto_strategy.genes.operators.crossover import (
            single_point_crossover,
        )

        generator = self._generator()
        base = generator.generate_random_gene()
        other = generator.generate_random_gene()

        for _ in range(30):
            child1, child2 = single_point_crossover(StrategyGene, base, other, _MTF_CFG)
            assert _dangling_operands(child1) == set()
            assert _dangling_operands(child2) == set()

            mutated = base.mutate(_MTF_CFG, mutation_rate=1.0)
            assert _dangling_operands(mutated) == set()
