"""
TPSLSettingsクラス

TP/SL設定を管理します。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .constants import TPSL_METHODS
from .base import BaseConfig

# GA TPSL関連定数
GA_DEFAULT_TPSL_METHOD_CONSTRAINTS = [
    "fixed_percentage",
    "risk_reward_ratio",
    "volatility_based",
    "statistical",
    "adaptive",
]

GA_TPSL_SL_RANGE = [0.01, 0.08]  # SL範囲（1%-8%）
GA_TPSL_TP_RANGE = [0.02, 0.20]  # TP範囲（2%-20%）
GA_TPSL_RR_RANGE = [1.2, 4.0]  # リスクリワード比範囲
GA_TPSL_ATR_MULTIPLIER_RANGE = [1.0, 4.0]  # ATR倍率範囲

# TPSL 制限設定
TPSL_LIMITS = {
    "stop_loss_pct": (0.005, 0.15),
    "take_profit_pct": (0.01, 0.3),
    "base_stop_loss": (0.005, 0.15),
    "atr_multiplier_sl": (0.5, 5.0),
    "atr_multiplier_tp": (1.0, 10.0),
    "atr_period": (5, 50),
    "lookback_period": (20, 500),
    "confidence_threshold": (0.1, 1.0),
}


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
