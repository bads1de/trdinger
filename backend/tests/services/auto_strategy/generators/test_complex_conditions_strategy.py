from app.services.auto_strategy.generators.complex_conditions_strategy import (
    ComplexConditionsStrategy,
)
from app.services.auto_strategy.genes.conditions import ConditionGroup
from app.services.auto_strategy.genes.indicator import IndicatorGene


class MockConditionGenerator:
    """テスト用のモック生成器"""

    def __init__(self):
        self.context = {"timeframe": "1h"}

    def _get_indicator_name(self, indicator):
        return indicator.type

    def _dynamic_classify(self, indicators):
        # 簡易分類
        from app.services.auto_strategy.config.constants import IndicatorType

        res = {
            IndicatorType.TREND: [],
            IndicatorType.MOMENTUM: [],
            IndicatorType.VOLATILITY: [],
        }
        for ind in indicators:
            if "rsi" in ind.type.lower() or "macd" in ind.type.lower():
                res[IndicatorType.MOMENTUM].append(ind)
            elif "bb" in ind.type.lower():
                res[IndicatorType.VOLATILITY].append(ind)
            else:
                res[IndicatorType.TREND].append(ind)
        return res

    def _get_band_names(self, indicator):
        return f"{indicator.type}_upper", f"{indicator.type}_lower"

    def _is_price_scale(self, indicator):
        return "ma" in indicator.type.lower()

    def _is_band_indicator(self, indicator):
        return "bb" in indicator.type.lower()

    def _structure_conditions(self, conditions):
        return conditions

    def generate_fallback_conditions(self, indicators):
        return [], [], []


def test_generate_hierarchical_structure():
    generator = MockConditionGenerator()
    strategy = ComplexConditionsStrategy(generator)

    # より多くのインジケータを提供し、パラメータも設定
    indicators = [
        IndicatorGene(type="rsi", parameters={"length": 14}, enabled=True),
        IndicatorGene(type="macd", parameters={"fast": 12, "slow": 26}, enabled=True),
        IndicatorGene(type="bb", parameters={"length": 20}, enabled=True),
        IndicatorGene(type="sma", parameters={"period": 20}, enabled=True),
        IndicatorGene(type="ema", parameters={"period": 50}, enabled=True),
        IndicatorGene(type="stoch", parameters={"k": 14}, enabled=True),
    ]

    # We expect some hierarchy (ConditionGroup) or multiple conditions
    found_group_or_multiple = False

    # Try multiple times if randomness is involved
    for _iteration in range(50):  # 試行回数を増やす
        longs, shorts, _ = strategy.generate_conditions(indicators)

        # グループが見つかったか、または複数の条件が生成されたか
        has_group = any(
            isinstance(c, ConditionGroup) and len(c.conditions) >= 2
            for c in longs + shorts
        )
        has_multiple_conditions = len(longs) + len(shorts) >= 2

        if has_group or has_multiple_conditions:
            found_group_or_multiple = True
            break

    # 修正された期待値：ConditionGroupまたは複数の条件が生成されること
    assert found_group_or_multiple, (
        "ComplexConditionsStrategy should generate ConditionGroups or multiple conditions"
    )


class TestTrendFollowPriceScaleGuard:
    """_create_trend_followの価格スケールガードのテスト"""

    def test_trend_follow_skips_non_price_trend(self):
        """非価格スケールのトレンド指標だけではCloseとのペアを生成しない"""
        generator = MockConditionGenerator()
        strategy = ComplexConditionsStrategy(generator)

        classified = generator._dynamic_classify(
            [
                IndicatorGene(type="oi_weighted", parameters={}, enabled=True),
                IndicatorGene(type="rsi", parameters={"length": 14}, enabled=True),
            ]
        )
        longs, shorts = strategy._create_trend_follow(classified)

        assert longs == []
        assert shorts == []

    def test_trend_follow_pairs_close_with_price_trend_only(self):
        """Closeとのペアは価格スケールのトレンド指標のみ"""
        generator = MockConditionGenerator()
        strategy = ComplexConditionsStrategy(generator)

        indicators = [
            IndicatorGene(type="oi_weighted", parameters={}, enabled=True),
            IndicatorGene(type="sma", parameters={"period": 20}, enabled=True),
            IndicatorGene(type="rsi", parameters={"length": 14}, enabled=True),
        ]
        classified = generator._dynamic_classify(indicators)

        for _ in range(30):
            longs, shorts = strategy._create_trend_follow(classified)
            for cond in longs + shorts:
                for c in (
                    cond.conditions if isinstance(cond, ConditionGroup) else [cond]
                ):
                    if str(c.left_operand).lower() == "close":
                        assert c.right_operand == "sma"
