"""
Settings クラス群

AutoStrategy の各種設定クラスを統合したモジュール。
TradingSettings, IndicatorSettings, TPSLSettings, PositionSizingSettings を提供します。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..utils.indicator_utils import get_valid_indicator_types
from ..utils.yaml_utils import YamlIndicatorUtils
from .base import BaseConfig
from .constants import (
    CONSTRAINTS,
    DATA_SOURCES,
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    GA_DEFAULT_POSITION_SIZING_METHOD_CONSTRAINTS,
    GA_DEFAULT_TPSL_METHOD_CONSTRAINTS,
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
    GA_TPSL_ATR_MULTIPLIER_RANGE,
    GA_TPSL_RR_RANGE,
    GA_TPSL_SL_RANGE,
    GA_TPSL_TP_RANGE,
    OPERATORS,
    POSITION_SIZING_LIMITS,
    POSITION_SIZING_METHODS,
    SUPPORTED_SYMBOLS,
    SUPPORTED_TIMEFRAMES,
    TPSL_LIMITS,
    TPSL_METHODS,
)


@dataclass
class TradingSettings(BaseConfig):
    """取引基本設定"""

    # 基本取引設定
    default_symbol: str = DEFAULT_SYMBOL
    default_timeframe: str = DEFAULT_TIMEFRAME
    supported_symbols: List[str] = field(
        default_factory=lambda: SUPPORTED_SYMBOLS.copy()
    )
    supported_timeframes: List[str] = field(
        default_factory=lambda: SUPPORTED_TIMEFRAMES.copy()
    )

    # 運用制約
    min_trades: int = CONSTRAINTS["min_trades"]
    max_drawdown_limit: float = CONSTRAINTS["max_drawdown_limit"]
    max_position_size: float = CONSTRAINTS["max_position_size"]
    min_position_size: float = CONSTRAINTS["min_position_size"]


@dataclass
class IndicatorSettings(BaseConfig):
    """テクニカル指標設定"""

    # 利用可能な指標
    valid_indicator_types: List[str] = field(default_factory=get_valid_indicator_types)

    # 指標特性データベース
    indicator_characteristics: Dict[str, Any] = field(
        default_factory=lambda: YamlIndicatorUtils.get_characteristics().copy()
    )

    # 演算子とデータソース
    operators: List[str] = field(default_factory=lambda: OPERATORS.copy())
    data_sources: List[str] = field(default_factory=lambda: DATA_SOURCES.copy())

    def get_all_indicators(self) -> List[str]:
        """全指標タイプを取得"""
        return self.valid_indicator_types

    def get_indicator_characteristics(self, indicator: str) -> Optional[Dict[str, Any]]:
        """特定の指標の特性を取得"""
        return self.indicator_characteristics.get(indicator)


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
