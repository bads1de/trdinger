"""
PositionSizingSettingsクラス

ポジションサイジング設定を管理します。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .base import BaseConfig
from ..constants import POSITION_SIZING_METHODS

# GA ポジションサイジング関連定数
GA_DEFAULT_POSITION_SIZING_METHOD_CONSTRAINTS = [
    "half_optimal_f",
    "volatility_based",
    "fixed_ratio",
    "fixed_quantity",
]

GA_POSITION_SIZING_LOOKBACK_RANGE = [50, 200]  # ハーフオプティマルF用ルックバック期間
GA_POSITION_SIZING_OPTIMAL_F_MULTIPLIER_RANGE = [0.25, 0.75]  # オプティマルF倍率範囲
GA_POSITION_SIZING_ATR_PERIOD_RANGE = [10, 30]  # ATR計算期間範囲
GA_POSITION_SIZING_ATR_MULTIPLIER_RANGE = [
    1.0,
    4.0,
]  # ポジションサイジング用ATR倍率範囲
GA_POSITION_SIZING_RISK_PER_TRADE_RANGE = [
    0.01,
    0.05,
]  # 1取引あたりのリスク範囲（1%-5%）
GA_POSITION_SIZING_FIXED_RATIO_RANGE = [0.05, 0.3]  # 固定比率範囲（5%-30%）
GA_POSITION_SIZING_FIXED_QUANTITY_RANGE = [0.1, 5.0]  # 固定枚数範囲
GA_POSITION_SIZING_MIN_SIZE_RANGE = [0.01, 0.1]  # 最小ポジションサイズ範囲
GA_POSITION_SIZING_MAX_SIZE_RANGE = [
    0.001,
    1.0,
]  # 最大ポジションサイズ範囲（システム全体のmax_position_sizeに一致）
GA_POSITION_SIZING_PRIORITY_RANGE = [0.5, 1.5]  # 優先度範囲
GA_POSITION_SIZING_VAR_CONFIDENCE_RANGE = [0.8, 0.99]  # VaR信頼水準
GA_POSITION_SIZING_MAX_VAR_RATIO_RANGE = [0.005, 0.05]  # VaR許容比率
GA_POSITION_SIZING_MAX_ES_RATIO_RANGE = [0.01, 0.1]  # ES許容比率
GA_POSITION_SIZING_VAR_LOOKBACK_RANGE = [50, 500]  # VaR計算のルックバック期間

# ポジションサイジング制限設定
POSITION_SIZING_LIMITS = {
    "lookback_period": (10, 500),
    "optimal_f_multiplier": (0.1, 1.0),
    "atr_period": (5, 50),
    "atr_multiplier": (0.5, 10.0),
    "risk_per_trade": (0.001, 0.1),
    "fixed_ratio": (0.01, 10.0),
    "fixed_quantity": (0.01, 1000.0),
    "min_position_size": (0.001, 1.0),
    "max_position_size": (0.001, 1.0),
    "var_confidence": (0.8, 0.999),
    "max_var_ratio": (0.001, 0.1),
    "max_expected_shortfall_ratio": (0.001, 0.2),
    "var_lookback": (20, 1000),
}


@dataclass
class PositionSizingSettings(BaseConfig):
    """ポジションサイジング設定"""

    # サイジング方法
    methods: List[str] = field(default_factory=lambda: POSITION_SIZING_METHODS.copy())
    default_methods: List[str] = field(
        default_factory=lambda: GA_DEFAULT_POSITION_SIZING_METHOD_CONSTRAINTS.copy()
    )

    # パラメータ範囲
    lookback_range: List[int] = field(
        default_factory=lambda: GA_POSITION_SIZING_LOOKBACK_RANGE.copy()
    )
    optimal_f_multiplier_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_OPTIMAL_F_MULTIPLIER_RANGE.copy()
    )
    atr_period_range: List[int] = field(
        default_factory=lambda: GA_POSITION_SIZING_ATR_PERIOD_RANGE.copy()
    )
    atr_multiplier_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_ATR_MULTIPLIER_RANGE.copy()
    )
    risk_per_trade_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_RISK_PER_TRADE_RANGE.copy()
    )
    fixed_ratio_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_FIXED_RATIO_RANGE.copy()
    )
    fixed_quantity_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_FIXED_QUANTITY_RANGE.copy()
    )
    min_size_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_MIN_SIZE_RANGE.copy()
    )
    max_size_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_MAX_SIZE_RANGE.copy()
    )
    priority_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_PRIORITY_RANGE.copy()
    )
    var_confidence_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_VAR_CONFIDENCE_RANGE.copy()
    )
    max_var_ratio_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_MAX_VAR_RATIO_RANGE.copy()
    )
    max_expected_shortfall_ratio_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_MAX_ES_RATIO_RANGE.copy()
    )
    var_lookback_range: List[int] = field(
        default_factory=lambda: GA_POSITION_SIZING_VAR_LOOKBACK_RANGE.copy()
    )

    # 制限設定
    limits: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: POSITION_SIZING_LIMITS.copy()
    )

    def get_default_values(self) -> Dict[str, Any]:
        """デフォルト値を取得（自動生成を利用）"""
        # フィールドから自動生成したデフォルト値を取得
        defaults = self.get_default_values_from_fields()
        # 必要に応じてカスタマイズ（外部定数など）
        return defaults
