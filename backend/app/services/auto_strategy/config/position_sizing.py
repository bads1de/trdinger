"""
PositionSizingSettingsクラス

ポジションサイジング設定を管理します。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .base import BaseConfig
from .constants import (
    GA_DEFAULT_POSITION_SIZING_METHOD_CONSTRAINTS,
    GA_POSITION_SIZING_ATR_MULTIPLIER_RANGE,
    GA_POSITION_SIZING_ATR_PERIOD_RANGE,
    GA_POSITION_SIZING_FIXED_QUANTITY_RANGE,
    GA_POSITION_SIZING_FIXED_RATIO_RANGE,
    GA_POSITION_SIZING_LOOKBACK_RANGE,
    GA_POSITION_SIZING_MAX_ES_RATIO_RANGE,
    GA_POSITION_SIZING_MAX_SIZE_RANGE,
    GA_POSITION_SIZING_MAX_VAR_RATIO_RANGE,
    GA_POSITION_SIZING_MIN_SIZE_RANGE,
    GA_POSITION_SIZING_OPTIMAL_F_MULTIPLIER_RANGE,
    GA_POSITION_SIZING_PRIORITY_RANGE,
    GA_POSITION_SIZING_RISK_PER_TRADE_RANGE,
    GA_POSITION_SIZING_VAR_CONFIDENCE_RANGE,
    GA_POSITION_SIZING_VAR_LOOKBACK_RANGE,
    POSITION_SIZING_LIMITS,
    POSITION_SIZING_METHODS,
)


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





