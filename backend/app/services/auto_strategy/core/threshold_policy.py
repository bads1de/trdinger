from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

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


class ThresholdPolicy:
    """指標閾値のプロファイル別ポリシー

    必要に応じて項目を追加可能。
    """

    DEFAULTS = {
        "aggressive": Thresholds(
            rsi_long_lt=51,
            rsi_short_gt=49,
            adx_trend_min=15,
            mfi_long_lt=48,
            mfi_short_gt=52,
            cci_abs_limit=120,
            willr_long_lt=52,  # -100〜0 を 0〜100 に線形変換した閾値相当
            willr_short_gt=48,
            ultosc_long_gt=52,
            ultosc_short_lt=48,
        ),
        "normal": Thresholds(
            rsi_long_lt=54,
            rsi_short_gt=46,
            adx_trend_min=18,
            mfi_long_lt=45,
            mfi_short_gt=55,
            cci_abs_limit=100,
            willr_long_lt=55,
            willr_short_gt=45,
            ultosc_long_gt=55,
            ultosc_short_lt=45,
        ),
        "conservative": Thresholds(
            rsi_long_lt=57,
            rsi_short_gt=43,
            adx_trend_min=22,
            mfi_long_lt=42,
            mfi_short_gt=58,
            cci_abs_limit=80,
            willr_long_lt=58,
            willr_short_gt=42,
            ultosc_long_gt=58,
            ultosc_short_lt=42,
        ),
    }

    @classmethod
    def get(cls, profile: Optional[Profile]) -> Thresholds:
        if profile is None:
            profile = "normal"  # type: ignore
        return cls.DEFAULTS.get(profile, cls.DEFAULTS["normal"])  # type: ignore
