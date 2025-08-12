import random
import numpy as np

from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator


def test_allowed_indicators_filter_is_respected():
    random.seed(456)
    np.random.seed(456)

    cfg = GAConfig(
        indicator_mode="technical_only",
        allowed_indicators=["SMA", "RSI"],
        max_indicators=3,
        min_indicators=2,
    )

    gen = RandomGeneGenerator(config=cfg, enable_smart_generation=True)
    gene = gen.generate_random_gene()

    assert gene.indicators, "インジケータが生成されていない"
    assert all(ind.type in {"SMA", "RSI"} for ind in gene.indicators), "allowed_indicators フィルタが遵守されていない"

