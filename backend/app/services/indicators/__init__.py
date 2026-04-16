"""
テクニカル指標パッケージ

分割されたテクニカル指標クラスと統合サービスを提供します。
新しいnumpy配列ベースのオートストラテジー最適化版も含みます。
"""

from .data_validation import (
    PandasTAError,
    handle_pandas_ta_errors,
    validate_data_length_with_fallback,
    validate_input,
    validate_multi_series_params,
    validate_series_params,
)
from .indicator_orchestrator import TechnicalIndicatorService
from .technical_indicators import (
    MomentumIndicators,
    OriginalIndicators,
    TrendIndicators,
    VolatilityIndicators,
    VolumeIndicators,
)

# 公開API
__all__ = [
    "TrendIndicators",
    "MomentumIndicators",
    "VolatilityIndicators",
    "VolumeIndicators",
    "OriginalIndicators",
    "PandasTAError",
    "validate_input",
    "validate_data_length_with_fallback",
    "validate_series_params",
    "validate_multi_series_params",
    "handle_pandas_ta_errors",
    "TechnicalIndicatorService",
]
