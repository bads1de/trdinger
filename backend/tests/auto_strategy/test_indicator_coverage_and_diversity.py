import random
import numpy as np

import pytest

from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.services.auto_strategy.models.gene_tpsl import TPSLMethod
from app.services.auto_strategy.models.gene_position_sizing import PositionSizingMethod
from app.services.indicators import TechnicalIndicatorService


@pytest.mark.parametrize("runs", [300])
def test_all_supported_indicators_can_appear_in_technical_only(runs):
    random.seed(777)
    np.random.seed(777)

    service = TechnicalIndicatorService()
    supported = set(service.get_supported_indicators().keys())

    # ML系は除外
    supported_technical = {name for name in supported if not name.startswith("ML_")}

    cfg = GAConfig(
        indicator_mode="technical_only",
        max_indicators=3,
        min_indicators=2,
        # すべてのサポート指標を許可し、フィルタを最小限に
        allowed_indicators=sorted(list(supported_technical)),
    )

    gen = RandomGeneGenerator(config=cfg, enable_smart_generation=True)

    seen = set()
    for i in range(runs):
        gene = gen.generate_random_gene()
        for ind in gene.indicators:
            if ind.enabled:
                seen.add(ind.type)

    uncovered = supported_technical - seen
    # 生成の確率性を考慮しても、ほぼ全てが登場できることを期待（残っても極小数）
    assert (
        len(uncovered) <= 2
    ), f"未出現が多すぎ: {sorted(list(uncovered))[:10]} (残={len(uncovered)})"


def test_tpsl_and_position_sizing_method_diversity_over_runs():
    random.seed(778)
    np.random.seed(778)

    cfg = GAConfig(indicator_mode="technical_only", max_indicators=3, min_indicators=2)
    gen = RandomGeneGenerator(config=cfg, enable_smart_generation=True)

    tpsl_methods_seen = set()
    ps_methods_seen = set()

    for _ in range(60):
        gene = gen.generate_random_gene()
        tpsl_methods_seen.add(gene.tpsl_gene.method)
        ps_methods_seen.add(gene.position_sizing_gene.method)

    # 多様性: 4種類中、最低3種類は出現
    assert len(tpsl_methods_seen) >= 3, f"TPSLメソッド多様性不足: {tpsl_methods_seen}"
    assert len(ps_methods_seen) >= 3, f"資金管理メソッド多様性不足: {ps_methods_seen}"
