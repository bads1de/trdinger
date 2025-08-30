from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal

from app.services.auto_strategy.models.strategy_models import IndicatorGene
from app.services.indicators.config import indicator_registry
from app.config.unified_config import unified_config


Profile = Literal["aggressive", "normal", "conservative"]


@dataclass(frozen=True)
class Thresholds:
    rsi_long_lt: int
    rsi_short_gt: int
    adx_trend_min: int
    mfi_long_lt: int | None = None
    mfi_short_gt: int | None = None
    cci_abs_limit: int | None = None
    willr_long_lt: int | None = None
    willr_short_gt: int | None = None
    ultosc_long_gt: int | None = None
    ultosc_short_lt: int | None = None
    roc_long_lt: int | None = None
    roc_short_gt: int | None = None
    mom_long_lt: float | None = None
    mom_short_gt: float | None = None
    stoch_long_lt: int | None = None
    stoch_short_gt: int | None = None
    cmo_long_lt: int | None = None
    cmo_short_gt: int | None = None
    trix_long_lt: float | None = None
    trix_short_gt: float | None = None
    bop_long_gt: float | None = None
    bop_short_lt: float | None = None
    apo_long_gt: float | None = None
    apo_short_lt: float | None = None


class ThresholdPolicy:
    """指標閾値のプロファイル別ポリシー

    必要に応じて項目を追加可能。
    """

    @property
    def DEFAULTS(self):
        """設定値から動的に閾値を生成"""
        return {
            "aggressive": Thresholds(
                rsi_long_lt=unified_config.indicators.rsi_long_lt_aggressive,
                rsi_short_gt=unified_config.indicators.rsi_short_gt_aggressive,
                adx_trend_min=unified_config.indicators.adx_trend_min_aggressive,
                mfi_long_lt=unified_config.indicators.mfi_long_lt_aggressive,
                mfi_short_gt=unified_config.indicators.mfi_short_gt_aggressive,
                cci_abs_limit=unified_config.indicators.cci_abs_limit_aggressive,
                willr_long_lt=unified_config.indicators.willr_long_lt_aggressive,
                willr_short_gt=unified_config.indicators.willr_short_gt_aggressive,
                ultosc_long_gt=unified_config.indicators.ultosc_long_gt_aggressive,
                ultosc_short_lt=unified_config.indicators.ultosc_short_lt_aggressive,
                roc_long_lt=unified_config.indicators.roc_long_lt_aggressive,
                roc_short_gt=unified_config.indicators.roc_short_gt_aggressive,
                mom_long_lt=unified_config.indicators.mom_long_lt_aggressive,
                mom_short_gt=unified_config.indicators.mom_short_gt_aggressive,
                stoch_long_lt=unified_config.indicators.stoch_long_lt_aggressive,
                stoch_short_gt=unified_config.indicators.stoch_short_gt_aggressive,
                cmo_long_lt=unified_config.indicators.cmo_long_lt_aggressive,
                cmo_short_gt=unified_config.indicators.cmo_short_gt_aggressive,
                trix_long_lt=unified_config.indicators.trix_long_lt_aggressive,
                trix_short_gt=unified_config.indicators.trix_short_gt_aggressive,
                bop_long_gt=unified_config.indicators.bop_long_gt_aggressive,
                bop_short_lt=unified_config.indicators.bop_short_lt_aggressive,
                apo_long_gt=unified_config.indicators.apo_long_gt_aggressive,
                apo_short_lt=unified_config.indicators.apo_short_lt_aggressive,
            ),
            "normal": Thresholds(
                rsi_long_lt=unified_config.indicators.rsi_long_lt_normal,
                rsi_short_gt=unified_config.indicators.rsi_short_gt_normal,
                adx_trend_min=unified_config.indicators.adx_trend_min_normal,
                mfi_long_lt=unified_config.indicators.mfi_long_lt_normal,
                mfi_short_gt=unified_config.indicators.mfi_short_gt_normal,
                cci_abs_limit=unified_config.indicators.cci_abs_limit_normal,
                willr_long_lt=unified_config.indicators.willr_long_lt_normal,
                willr_short_gt=unified_config.indicators.willr_short_gt_normal,
                ultosc_long_gt=unified_config.indicators.ultosc_long_gt_normal,
                ultosc_short_lt=unified_config.indicators.ultosc_short_lt_normal,
                roc_long_lt=unified_config.indicators.roc_long_lt_normal,
                roc_short_gt=unified_config.indicators.roc_short_gt_normal,
                mom_long_lt=unified_config.indicators.mom_long_lt_normal,
                mom_short_gt=unified_config.indicators.mom_short_gt_normal,
                stoch_long_lt=unified_config.indicators.stoch_long_lt_normal,
                stoch_short_gt=unified_config.indicators.stoch_short_gt_normal,
                cmo_long_lt=unified_config.indicators.cmo_long_lt_normal,
                cmo_short_gt=unified_config.indicators.cmo_short_gt_normal,
                trix_long_lt=unified_config.indicators.trix_long_lt_normal,
                trix_short_gt=unified_config.indicators.trix_short_gt_normal,
                bop_long_gt=unified_config.indicators.bop_long_gt_normal,
                bop_short_lt=unified_config.indicators.bop_short_lt_normal,
                apo_long_gt=unified_config.indicators.apo_long_gt_normal,
                apo_short_lt=unified_config.indicators.apo_short_lt_normal,
            ),
            "conservative": Thresholds(
                rsi_long_lt=unified_config.indicators.rsi_long_lt_conservative,
                rsi_short_gt=unified_config.indicators.rsi_short_gt_conservative,
                adx_trend_min=unified_config.indicators.adx_trend_min_conservative,
                mfi_long_lt=unified_config.indicators.mfi_long_lt_conservative,
                mfi_short_gt=unified_config.indicators.mfi_short_gt_conservative,
                cci_abs_limit=unified_config.indicators.cci_abs_limit_conservative,
                willr_long_lt=unified_config.indicators.willr_long_lt_conservative,
                willr_short_gt=unified_config.indicators.willr_short_gt_conservative,
                ultosc_long_gt=unified_config.indicators.ultosc_long_gt_conservative,
                ultosc_short_lt=unified_config.indicators.ultosc_short_lt_conservative,
                roc_long_lt=unified_config.indicators.roc_long_lt_conservative,
                roc_short_gt=unified_config.indicators.roc_short_gt_conservative,
                mom_long_lt=unified_config.indicators.mom_long_lt_conservative,
                mom_short_gt=unified_config.indicators.mom_short_gt_conservative,
                stoch_long_lt=unified_config.indicators.stoch_long_lt_conservative,
                stoch_short_gt=unified_config.indicators.stoch_short_gt_conservative,
                cmo_long_lt=unified_config.indicators.cmo_long_lt_conservative,
                cmo_short_gt=unified_config.indicators.cmo_short_gt_conservative,
                trix_long_lt=unified_config.indicators.trix_long_lt_conservative,
                trix_short_gt=unified_config.indicators.trix_short_gt_conservative,
                bop_long_gt=unified_config.indicators.bop_long_gt_conservative,
                bop_short_lt=unified_config.indicators.bop_short_lt_conservative,
                apo_long_gt=unified_config.indicators.apo_long_gt_conservative,
                apo_short_lt=unified_config.indicators.apo_short_lt_conservative,
            ),
        }

    @classmethod
    def get(cls, profile: Optional[Profile]) -> Thresholds:
        if profile is None:
            profile = "normal"  # type: ignore
        defaults = cls().DEFAULTS  # Create instance to access property
        return defaults.get(profile, defaults["normal"])  # type: ignore


TREND_PREF = (
    "SMA",
    "EMA",
    "MA",
    "HMA",
    "WMA",
    "RMA",
    "HT_TRENDLINE",
    "FWMA",
    "SWMA",
    "VIDYA",
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
