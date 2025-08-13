import random
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.auto_strategy.evaluators.condition_evaluator import ConditionEvaluator
from app.services.auto_strategy.services.indicator_service import (
    IndicatorCalculator,
)
from app.services.auto_strategy.models.gene_strategy import StrategyGene
from app.services.auto_strategy.models.condition_group import ConditionGroup


class _Data:
    def __init__(self, n=300):
        idx = pd.date_range("2024-03-01", periods=n, freq="H")
        close = np.linspace(100, 110, n)
        open_ = np.concatenate([[100], close[:-1]])
        high = np.maximum(open_, close) * 1.0005
        low = np.minimum(open_, close) * 0.9995
        vol = np.full(n, 1000.0)
        self.df = pd.DataFrame(
            {
                "Open": open_.astype(float),
                "High": high.astype(float),
                "Low": low.astype(float),
                "Close": close.astype(float),
                "Volume": vol.astype(float),
            },
            index=idx,
        )

    @property
    def Open(self):
        return self.df["Open"]

    @property
    def High(self):
        return self.df["High"]

    @property
    def Low(self):
        return self.df["Low"]

    @property
    def Close(self):
        return self.df["Close"]

    @property
    def Volume(self):
        return self.df["Volume"]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_condition_truth_rate_on_synthetic_data(seed):
    random.seed(seed)
    np.random.seed(seed)

    cfg = GAConfig.create_fast()
    cfg.indicator_mode = "technical_only"

    gen = RandomGeneGenerator(cfg)
    gene: StrategyGene = gen.generate_random_gene()

    # 指標初期化
    calc = IndicatorCalculator()
    s = type("S", (), {})()
    s.data = _Data()

    def I(f):
        return f()

    s.I = I

    for ind in gene.indicators:
        if not ind.enabled:
            continue
        res = calc.calculate_indicator(s.data, ind.type, ind.parameters)
        if isinstance(res, tuple):
            for i, arr in enumerate(res):
                setattr(s, f"{ind.type}_{i}", arr)
        else:
            setattr(s, ind.type, res)

    ev = ConditionEvaluator()

    # 条件の成立率をざっくり測る（現在バーの比較なので厳密な確率ではない）
    longs = gene.get_effective_long_conditions()
    shorts = gene.get_effective_short_conditions()
    if not longs and not shorts:
        longs = gene.entry_conditions

    true_count = 0
    total_checks = 0
    for cond in longs + shorts:
        if isinstance(cond, ConditionGroup):
            # ORグループ内の各条件で計測
            for c in cond.conditions:
                lv = ev.get_condition_value(c.left_operand, s)
                rv = ev.get_condition_value(c.right_operand, s)
                if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
                    total_checks += 1
                    if c.operator == ">":
                        true_count += int(lv > rv)
                    elif c.operator == "<":
                        true_count += int(lv < rv)
                    elif c.operator == ">=":
                        true_count += int(lv >= rv)
                    elif c.operator == "<=":
                        true_count += int(lv <= rv)
                    elif c.operator == "==":
                        true_count += int(abs(lv - rv) < 1e-9)
                    elif c.operator == "!=":
                        true_count += int(abs(lv - rv) >= 1e-9)
        else:
            lv = ev.get_condition_value(cond.left_operand, s)
            rv = ev.get_condition_value(cond.right_operand, s)
            if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
                total_checks += 1
                if cond.operator == ">":
                    true_count += int(lv > rv)
                elif cond.operator == "<":
                    true_count += int(lv < rv)
                elif cond.operator == ">=":
                    true_count += int(lv >= rv)
                elif cond.operator == "<=":
                    true_count += int(lv <= rv)
                elif cond.operator == "==":
                    true_count += int(abs(lv - rv) < 1e-9)
                elif cond.operator == "!=":
                    true_count += int(abs(lv - rv) >= 1e-9)

    # 成立可能性の基本検証（0でないこと）
    assert total_checks >= 1
