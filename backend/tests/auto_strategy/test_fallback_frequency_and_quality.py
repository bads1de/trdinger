import random
import numpy as np

from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)



def test_fallback_strategy_generation_is_rare():
    random.seed(779)
    np.random.seed(779)

    cfg = GAConfig(indicator_mode="technical_only", max_indicators=3, min_indicators=2)
    gen = RandomGeneGenerator(config=cfg, enable_smart_generation=True)

    count = 0
    for _ in range(200):
        gene = gen.generate_random_gene()
        # フォールバックは metadata.generated_by で判定
        if (
            gene.metadata
            and gene.metadata.get("generated_by") == "default_gene_utility"
        ):
            count += 1

    # フォールバック戦略の発生率は 5% 未満を期待（例外などの頻度が低いこと）
    assert count <= 10, f"フォールバック発生が多すぎる: {count}/200"
