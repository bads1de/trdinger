import random
import numpy as np

from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.services.auto_strategy.models.condition_group import ConditionGroup


def test_or_group_presence_and_condition_mix():
    random.seed(780)
    np.random.seed(780)

    cfg = GAConfig(indicator_mode="technical_only", max_indicators=3, min_indicators=2)
    gen = RandomGeneGenerator(config=cfg, enable_smart_generation=True)

    or_present = 0
    price_vs_trend_present = 0

    for _ in range(50):
        gene = gen.generate_random_gene()

        def has_price_vs_trend(conds):
            return any(
                (
                    c.left_operand in ("close", "open")
                    and isinstance(c.right_operand, str)
                    and c.right_operand in ("SMA", "EMA", "MAMA", "MA")
                )
                for c in conds
                if not isinstance(c, ConditionGroup)
            )

        if any(
            isinstance(c, ConditionGroup)
            for c in gene.long_entry_conditions + gene.short_entry_conditions
        ):
            or_present += 1

        if has_price_vs_trend(gene.long_entry_conditions) or has_price_vs_trend(
            gene.short_entry_conditions
        ):
            price_vs_trend_present += 1

    assert or_present >= 10, "ORグループの出現率が低すぎる"
    assert price_vs_trend_present >= 25, "価格vsトレンド条件の存在率が低すぎる"
