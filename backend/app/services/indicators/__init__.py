"""
テクニカル指標パッケージ

分割されたテクニカル指標クラスと統合サービスを提供します。
新しいnumpy配列ベースのオートストラテジー最適化版も含みます。
"""

from .indicator_orchestrator import TechnicalIndicatorService

from .technical_indicators.momentum import MomentumIndicators
from .technical_indicators.trend import TrendIndicators
from .technical_indicators.volatility import VolatilityIndicators
from .technical_indicators.volume import VolumeIndicators
from .utils import (
    PandasTAError,
    validate_input,
)
from .data_validation import (
    validate_data_length,
    validate_data_length_with_fallback,
    validate_ohlcv_data_quality,
    validate_indicator_params,
)

# 公開API
__all__ = [
    "TrendIndicators",
    "MomentumIndicators",
    "VolatilityIndicators",
    "VolumeIndicators",
    "PandasTAError",
    "validate_input",
    "validate_data_length",
    "validate_data_length_with_fallback",
    "validate_ohlcv_data_quality",
    "validate_indicator_params",
    "TechnicalIndicatorService",
]
