import numpy as np

from app.services.auto_strategy.core.indicator_name_resolver import IndicatorNameResolver
from app.services.auto_strategy.core.threshold_policy import ThresholdPolicy
from app.services.auto_strategy.core.price_trend_policy import PriceTrendPolicy
from app.services.auto_strategy.core.condition_assembly import ConditionAssembly
from app.services.auto_strategy.models.gene_strategy import IndicatorGene, Condition


class _S:
    def __init__(self):
        self.data = type("D", (), {
            "Close": np.array([1, 2, 3, 4, 5], dtype=float),
        })()
        self.SMA = np.array([1, 2, 3, 4, 5], dtype=float)
        self.MACD_0 = np.array([np.nan, 0, 1, 2, 3], dtype=float)
        self.BB_1 = np.array([10, 11, 12, 13, 14], dtype=float)


def test_threshold_policy_profiles():
    assert ThresholdPolicy.get("aggressive").rsi_long_lt < ThresholdPolicy.get("normal").rsi_long_lt
    assert ThresholdPolicy.get("conservative").adx_trend_min > ThresholdPolicy.get("normal").adx_trend_min


def test_price_trend_policy_pick():
    inds = [IndicatorGene(type="RSI"), IndicatorGene(type="EMA")]
    chosen = PriceTrendPolicy.pick_trend_name(inds)
    assert chosen in ("SMA", "EMA", "MAMA", "MA", "HMA", "WMA", "RMA", "HT_TRENDLINE")


def test_condition_assembly_or_and_fallback():
    inds = [IndicatorGene(type="EMA")]
    conds = [Condition(left_operand="RSI", operator="<", right_operand=30)]
    out = ConditionAssembly.ensure_or_with_fallback(conds, "long", inds)
    # RSI条件と、価格vsトレンド（EMA優先）が入るはず
    flat = []
    from app.services.auto_strategy.models.condition_group import ConditionGroup

    for c in out:
        if isinstance(c, ConditionGroup):
            flat.extend(c.conditions)
        else:
            flat.append(c)
    assert any(c.left_operand == "close" and isinstance(c.right_operand, str) for c in flat)


def test_indicator_name_resolver_integration():
    s = _S()
    ok, v = IndicatorNameResolver.try_resolve_value("BB_Middle_20", s)
    assert ok and v == 14
    ok, v = IndicatorNameResolver.try_resolve_value("MACD", s)
    assert ok and v == 3

