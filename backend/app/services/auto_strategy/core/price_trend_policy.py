from __future__ import annotations

from typing import List, Optional

from app.services.auto_strategy.models.gene_strategy import IndicatorGene
from app.services.indicators.config import indicator_registry


TREND_PREF = (
    "SMA",
    "EMA",
    "MA",
    "HMA",
    "WMA",
    "RMA",
    "HT_TRENDLINE",
)  # MAMA除外: 条件右オペランド未サポート


class PriceTrendPolicy:
    """price vs trend 条件用のトレンド候補選定ポリシー"""

    @staticmethod
    def pick_trend_name(indicators: List[IndicatorGene]) -> Optional[str]:
        # geneに含まれる有効なトレンド系から優先的に選択
        trend_pool = []
        for ind in indicators or []:
            if not getattr(ind, "enabled", True):
                continue
            cfg = indicator_registry.get_indicator_config(ind.type)
            if cfg and getattr(cfg, "category", None) == "trend":
                trend_pool.append(ind.type)
        # 優先候補
        pref = [n for n in trend_pool if n in TREND_PREF]
        if pref:
            import random

            return random.choice(pref)
        if trend_pool:
            import random

            return random.choice(trend_pool)
        # geneに無い場合はフォールバックでMA系から選択
        import random

        return random.choice(TREND_PREF)
