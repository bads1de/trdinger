"""
TPSLSettingsクラス

TP/SL設定を管理します。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .base import BaseConfig
from ..constants import (
    TPSL_METHODS,
    TPSL_LIMITS,
    GA_DEFAULT_TPSL_METHOD_CONSTRAINTS,
    GA_TPSL_SL_RANGE,
    GA_TPSL_TP_RANGE,
    GA_TPSL_RR_RANGE,
    GA_TPSL_ATR_MULTIPLIER_RANGE,
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

    def get_default_values(self) -> Dict[str, Any]:
        """デフォルト値を取得（自動生成を利用）"""
        # フィールドから自動生成したデフォルト値を取得
        defaults = self.get_default_values_from_fields()
        # 必要に応じてカスタマイズ（外部定数など）
        return defaults

    def get_limits_for_param(self, param_name: str) -> Tuple[float, float]:
        """指定されたパラメータの制限を取得"""
        if param_name in self.limits:
            return self.limits[param_name]
        raise ValueError(f"不明なパラメータ: {param_name}")