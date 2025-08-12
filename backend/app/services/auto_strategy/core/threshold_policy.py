from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Profile = Literal["aggressive", "normal", "conservative"]


@dataclass(frozen=True)
class Thresholds:
    rsi_long_lt: int
    rsi_short_gt: int
    adx_trend_min: int


class ThresholdPolicy:
    """指標閾値のプロファイル別ポリシー

    必要に応じて項目を追加可能。
    """

    DEFAULTS = {
        "aggressive": Thresholds(rsi_long_lt=51, rsi_short_gt=49, adx_trend_min=15),
        "normal": Thresholds(rsi_long_lt=54, rsi_short_gt=46, adx_trend_min=18),
        "conservative": Thresholds(rsi_long_lt=57, rsi_short_gt=43, adx_trend_min=22),
    }

    @classmethod
    def get(cls, profile: Optional[Profile]) -> Thresholds:
        if profile is None:
            profile = "normal"  # type: ignore
        return cls.DEFAULTS.get(profile, cls.DEFAULTS["normal"])  # type: ignore

