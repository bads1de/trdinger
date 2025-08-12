import random
import numpy as np

from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator


def test_technical_only_tpsl_enabled_and_exit_empty():
    random.seed(123)
    np.random.seed(123)

    cfg = GAConfig(
        indicator_mode="technical_only",
        max_indicators=3,
        min_indicators=2,
        max_conditions=4,
        min_conditions=2,
    )

    gen = RandomGeneGenerator(config=cfg, enable_smart_generation=True)
    gene = gen.generate_random_gene()

    assert gene.tpsl_gene is not None, "TP/SL 遺伝子が生成されていない"
    assert gene.tpsl_gene.enabled is True, "Auto-StrategyではTP/SLは常時有効化されるべき"
    assert gene.exit_conditions == [], "TP/SL有効時はexit条件は空であるべき"

