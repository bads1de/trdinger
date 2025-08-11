import random
import numpy as np
import pytest

from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.services.auto_strategy.generators.smart_condition_generator import (
    SmartConditionGenerator,
)
from app.services.auto_strategy.models.gene_strategy import IndicatorGene, Condition
from app.services.auto_strategy.models.condition_group import ConditionGroup
from app.services.indicators.config import indicator_registry


def test_random_gene_generator_technical_only_generates_strategy():
    # 再現性のためシード固定
    random.seed(42)
    np.random.seed(42)

    # テクニカルオンリーモードでGAConfigを作成
    cfg = GAConfig(
        indicator_mode="technical_only",
        max_indicators=3,
        min_indicators=2,
        max_conditions=4,
        min_conditions=2,
    )

    gen = RandomGeneGenerator(config=cfg, enable_smart_generation=True)

    gene = gen.generate_random_gene()

    # インジケータが1つ以上生成され、ML系が含まれないこと
    assert gene.indicators, "少なくとも1つのインジケータが必要です"
    assert all(
        not ind.type.startswith("ML_") for ind in gene.indicators
    ), "ML系は含めない"

    # ロング/ショート条件が生成されること（どちらかは1件以上）
    assert isinstance(gene.long_entry_conditions, list)
    assert isinstance(gene.short_entry_conditions, list)
    assert len(gene.long_entry_conditions) + len(gene.short_entry_conditions) >= 1

    # TP/SL遺伝子が有効な場合、exit_conditionsは空である（仕様準拠）
    if gene.tpsl_gene and gene.tpsl_gene.enabled:
        assert gene.exit_conditions == []


def test_smart_condition_generator_with_basic_technical_indicators():
    # レジストリから利用可能なテクニカル指標を選択（SMA/RSI/BBがあれば優先）
    candidates = []
    for name, params in [
        ("SMA", {"period": 20}),
        ("RSI", {"period": 14}),
        ("BB", {"period": 20}),
    ]:
        if indicator_registry.get_indicator_config(name) is not None:
            candidates.append(IndicatorGene(type=name, parameters=params, enabled=True))

    if len(candidates) < 2:
        pytest.skip("テクニカル指標が不足のためスキップ")

    gen = SmartConditionGenerator(enable_smart_generation=True)
    long_conds, short_conds, exit_conds = gen.generate_balanced_conditions(candidates)

    # 条件の基本検証
    for lst in (long_conds, short_conds, exit_conds):
        assert isinstance(lst, list)
        for c in lst:
            if isinstance(c, ConditionGroup):
                for sc in c.conditions:
                    assert sc.operator in {
                        ">",
                        "<",
                        ">=",
                        "<=",
                        "==",
                        "!=",
                        "above",
                        "below",
                    }
                    assert isinstance(sc.left_operand, (str, float, dict))
                    assert isinstance(sc.right_operand, (str, float, dict))
            else:
                assert isinstance(c, Condition)
                assert c.operator in {
                    ">",
                    "<",
                    ">=",
                    "<=",
                    "==",
                    "!=",
                    "above",
                    "below",
                }
                assert isinstance(c.left_operand, (str, float, dict))
                assert isinstance(c.right_operand, (str, float, dict))

    # ロングかショートのどちらかには最低1つ生成されること
    assert len(long_conds) + len(short_conds) >= 1
