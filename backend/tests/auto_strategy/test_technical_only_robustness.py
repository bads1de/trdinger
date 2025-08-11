import random
import numpy as np
import pytest

from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.services.auto_strategy.models.gene_serialization import GeneSerializer


def test_technical_only_random_generation_robustness_runs_30_times():
    serializer = GeneSerializer()

    for seed in range(1000, 1030):  # 30回
        random.seed(seed)
        np.random.seed(seed)

        cfg = GAConfig(
            indicator_mode="technical_only",
            max_indicators=3,
            min_indicators=2,
            max_conditions=4,
            min_conditions=2,
        )

        gen = RandomGeneGenerator(config=cfg, enable_smart_generation=True)

        # 生成が例外なく行えること
        strategy = gen.generate_random_gene()

        # シリアライズも例外なく行えること
        json_str = serializer.strategy_gene_to_json(strategy)
        assert isinstance(json_str, str) and len(json_str) > 0

        # 基本妥当性
        assert strategy.indicators, "インジケータが空"
        assert all(not ind.type.startswith("ML_") for ind in strategy.indicators)
        assert isinstance(strategy.long_entry_conditions, list)
        assert isinstance(strategy.short_entry_conditions, list)
        assert len(strategy.long_entry_conditions) + len(strategy.short_entry_conditions) >= 1

        # TP/SL有効時はexit条件が空
        if strategy.tpsl_gene and strategy.tpsl_gene.enabled:
            assert strategy.exit_conditions == []

