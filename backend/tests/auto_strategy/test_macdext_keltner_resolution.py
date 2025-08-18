import numpy as np
import pandas as pd
import pytest

from app.services.auto_strategy.services.indicator_service import IndicatorCalculator
from app.services.auto_strategy.core.condition_evaluator import ConditionEvaluator
from app.services.auto_strategy.models.gene_strategy import Condition, IndicatorGene
from app.services.auto_strategy.generators.smart_condition_generator import (
    SmartConditionGenerator,
)


class _FakeStrategy:
    def __init__(self, df: pd.DataFrame):
        class _D:
            pass

        self.data = _D()
        # backtesting.py 互換: data に大文字カラムがあるDataFrameをぶら下げる
        self.data.df = df


def _make_df(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="H")
    base = np.linspace(100, 110, n)
    df = pd.DataFrame(
        {
            "Open": base,
            "High": base * 1.001,
            "Low": base * 0.999,
            "Close": base + np.sin(np.linspace(0, 10, n)),
            "Volume": 1000,
        },
        index=idx,
    )
    return df


def test_condition_evaluator_resolves_macdext_and_keltner_components():
    df = _make_df()
    calc = IndicatorCalculator()
    s = _FakeStrategy(df)

    # MACDEXT (3 outputs) -> _0: main, _1: signal, _2: hist
    macdext = calc.calculate_indicator(
        s.data,
        "MACDEXT",
        {
            "fast_period": 12,
            "fast_ma_type": 0,
            "slow_period": 26,
            "slow_ma_type": 0,
            "signal_period": 9,
            "signal_ma_type": 0,
        },
    )
    assert isinstance(macdext, tuple) and len(macdext) == 3
    s.MACDEXT_0, s.MACDEXT_1, s.MACDEXT_2 = macdext

    # KELTNER (3 outputs) -> (upper, middle, lower)
    upper, middle, lower = calc.technical_indicator_service.calculate_indicator(
        df, "KELTNER", {"period": 20, "scalar": 2.0}
    )
    s.KELTNER_0, s.KELTNER_1, s.KELTNER_2 = upper, middle, lower

    ev = ConditionEvaluator()

    # 基本名解決: MACDEXT -> MACDEXT_0, KELTNER -> KELTNER_1
    val_macdext = ev.get_condition_value("MACDEXT", s)
    val_keltner = ev.get_condition_value("KELTNER", s)

    assert isinstance(val_macdext, float)
    assert isinstance(val_keltner, float)

    # 末尾値が有限
    assert np.isfinite(val_macdext)
    assert np.isfinite(val_keltner)


def test_smart_generator_handles_macdext_zero_center():
    gen = SmartConditionGenerator(enable_smart_generation=True)

    # 登録済みの指標遺伝子として MACDEXT を渡す
    ind = IndicatorGene(
        type="MACDEXT",
        parameters={
            "fast_period": 12,
            "fast_ma_type": 0,
            "slow_period": 26,
            "slow_ma_type": 0,
            "signal_period": 9,
            "signal_ma_type": 0,
        },
        enabled=True,
    )

    longs, shorts, _ = gen.generate_balanced_conditions([ind])

    # いずれかに MACDEXT 系のゼロ基準比較が含まれる可能性を検証
    def _contains_zero_center_check(conds):
        for c in conds:
            if (
                isinstance(c, Condition)
                and isinstance(c.left_operand, str)
                and (
                    c.left_operand.startswith("MACDEXT") or c.left_operand == "MACDEXT"
                )
                and isinstance(c.right_operand, (int, float))
                and c.right_operand == 0
            ):
                return True
        return False

    assert _contains_zero_center_check(longs) or _contains_zero_center_check(shorts)
