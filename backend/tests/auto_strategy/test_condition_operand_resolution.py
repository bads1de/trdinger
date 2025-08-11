import random
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import pytest

from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.auto_strategy.models.gene_strategy import StrategyGene, Condition
from app.services.auto_strategy.models.condition_group import ConditionGroup
from app.services.auto_strategy.calculators.indicator_calculator import (
    IndicatorCalculator,
)
from app.services.auto_strategy.evaluators.condition_evaluator import ConditionEvaluator


class _FakeData:
    def __init__(self, n=300):
        idx = pd.date_range("2024-01-01", periods=n, freq="H")
        close = np.linspace(100, 120, n)
        open_ = np.concatenate([[100], close[:-1]])
        high = np.maximum(open_, close) * 1.001
        low = np.minimum(open_, close) * 0.999
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

    # backtesting.py互換のプロパティアクセスを提供
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


class _FakeStrategy:
    def __init__(self, data: _FakeData):
        self.data = data
        # I() は与えられた関数の戻り値（配列）をそのまま返す簡易実装

    def I(self, func):
        return func()


def _init_indicators_for_gene(gene: StrategyGene, strategy) -> List[str]:
    calc = IndicatorCalculator()
    registered: List[str] = []
    for ind in gene.indicators:
        if not ind.enabled:
            continue
        result = calc.calculate_indicator(strategy.data, ind.type, ind.parameters)
        # IndicatorCalculator.init_indicator() と同じ登録規約で属性を付与
        if isinstance(result, tuple):
            for i, output in enumerate(result):
                name = f"{ind.type}_{i}"
                setattr(strategy, name, output)
                registered.append(name)
        else:
            setattr(strategy, ind.type, result)
            registered.append(ind.type)
    return registered


@pytest.mark.parametrize("seed", [42, 123, 987])
def test_random_condition_operands_resolve(seed):
    random.seed(seed)
    np.random.seed(seed)

    cfg = GAConfig.create_fast()
    cfg.indicator_mode = "technical_only"  # 根本原因の切り分け（MLを除外）

    gen = RandomGeneGenerator(cfg)
    gene = gen.generate_random_gene()

    strategy = _FakeStrategy(_FakeData(n=300))
    registered = _init_indicators_for_gene(gene, strategy)

    evaluator = ConditionEvaluator()

    # 有効なロング・ショート条件をまとめて検査
    conds: List[Condition] = []
    conds.extend(gene.get_effective_long_conditions())
    conds.extend(gene.get_effective_short_conditions())
    if not conds:
        conds.extend(gene.entry_conditions)

    # オペランド解決状況を収集
    unresolved: List[Tuple[str, Any]] = []
    resolved: List[Tuple[str, float]] = []

    for c in conds:
        # ConditionGroupなら子条件を展開
        if isinstance(c, ConditionGroup):
            subconds = c.conditions
        else:
            subconds = [c]
        for sc in subconds:
            for side in (sc.left_operand, sc.right_operand):
                try:
                    val = evaluator.get_condition_value(side, strategy)
                    if isinstance(side, str):
                        # 文字列オペランドで属性未解決の可能性を重点チェック
                        has_attr = hasattr(strategy, side)
                        mapped = False
                        if isinstance(side, str) and "_" in side:
                            base = side.split("_")[0]
                            mapped = hasattr(strategy, base)
                        if (
                            (not has_attr)
                            and (not mapped)
                            and val == 0.0
                            and side.lower()
                            not in {"open", "high", "low", "close", "volume"}
                        ):
                            unresolved.append((side, val))
                        else:
                            resolved.append((side, val))
                    else:
                        resolved.append((str(side), float(val)))
                except Exception as e:
                    unresolved.append((f"{side}", f"EXC:{e}"))

    # 診断出力（-s向け）
    print(
        {
            "registered_attrs": registered[:10],
            "n_registered": len(registered),
            "n_conditions": len(conds),
            "resolved": len(resolved),
            "unresolved": len(unresolved),
            "sample_unresolved": unresolved[:5],
        }
    )

    # 最低限の不変条件: 何らかの指標属性は登録され、条件オペランドの過半は解決される
    assert len(registered) > 0
    assert len(resolved) >= len(unresolved)
