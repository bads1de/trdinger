"""
TP/SL設定

TPSLSettings クラスを提供します。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .base import BaseConfig
from .constants import (
    GA_DEFAULT_TPSL_METHOD_CONSTRAINTS,
    GA_TPSL_ATR_MULTIPLIER_RANGE,
    GA_TPSL_RR_RANGE,
    GA_TPSL_SL_RANGE,
    GA_TPSL_TP_RANGE,
    TPSL_LIMITS,
    TPSL_METHODS,
)


@dataclass
class TPSLSettings(BaseConfig):
    """TP/SL設定"""

    # TPSL方法
    methods: List[str] = field(default_factory=lambda: TPSL_METHODS.copy())
    default_tpsl_methods: List[str] = field(
        default_factory=lambda: GA_DEFAULT_TPSL_METHOD_CONSTRAINTS.copy()
    )

    # パラメータ範囲
    sl_range: List[float] = field(default_factory=lambda: GA_TPSL_SL_RANGE.copy())
    tp_range: List[float] = field(default_factory=lambda: GA_TPSL_TP_RANGE.copy())
    rr_range: List[float] = field(default_factory=lambda: GA_TPSL_RR_RANGE.copy())
    atr_multiplier_range: List[float] = field(
        default_factory=lambda: GA_TPSL_ATR_MULTIPLIER_RANGE.copy()
    )

    # 制限設定
    limits: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: TPSL_LIMITS.copy()
    )
