"""
テクニカル指標パッケージ

分割されたテクニカル指標クラスと統合サービスを提供します。
新しいnumpy配列ベースのオートストラテジー最適化版も含みます。
"""

from .data_validation import (
    PandasTAError,
    validate_data_length_with_fallback,
    validate_input,
    validate_series_params,
    validate_multi_series_params,
    handle_pandas_ta_errors,
)
from .indicator_orchestrator import TechnicalIndicatorService
from .technical_indicators.momentum import MomentumIndicators
from .technical_indicators.trend import TrendIndicators
from .technical_indicators.volatility import VolatilityIndicators
from .technical_indicators.volume import VolumeIndicators

# 公開API
__all__ = [
    "TrendIndicators",
    "MomentumIndicators",
    "VolatilityIndicators",
    "VolumeIndicators",
    "PandasTAError",
    "validate_input",
    "validate_data_length_with_fallback",
    "validate_series_params",
    "validate_multi_series_params",
    "handle_pandas_ta_errors",
    "TechnicalIndicatorService",
]
