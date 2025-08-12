import random
import numpy as np
import pandas as pd

from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator


def test_random_gene_generator_condition_count_in_range():
    # 固定シードで安定化
    random.seed(123)
    np.random.seed(123)

    cfg = GAConfig(
        indicator_mode="technical_only",
        max_indicators=5,
        min_indicators=3,
        max_conditions=7,
        min_conditions=3,
    )

    gene = RandomGeneGenerator(config=cfg, enable_smart_generation=True).generate_random_gene()

    # 後方互換entry_conditionsも生成される
    assert 3 <= len(gene.entry_conditions) <= 7

    # long/short 条件はSmartConditionGenerator依存で可変（ここでは存在のみ）
    assert isinstance(gene.long_entry_conditions, list)
    assert isinstance(gene.short_entry_conditions, list)

