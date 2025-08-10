"""
テクニカル指標パッケージ

分割されたテクニカル指標クラスと統合サービスを提供します。
新しいnumpy配列ベースのオートストラテジー最適化版も含みます。
"""

# 既存のクラス（互換性維持）
from .indicator_orchestrator import TechnicalIndicatorService

from .technical_indicators.math_operators import MathOperatorsIndicators
from .technical_indicators.math_transform import MathTransformIndicators
from .technical_indicators.momentum import MomentumIndicators
from .technical_indicators.pattern_recognition import PatternRecognitionIndicators
from .technical_indicators.price_transform import PriceTransformIndicators
from .technical_indicators.statistics import StatisticsIndicators
from .technical_indicators.trend import TrendIndicators
from .technical_indicators.volatility import VolatilityIndicators
from .technical_indicators.volume import VolumeIndicators
from .utils import (
    PandasTAError,
    validate_input,
    validate_series_data,
    validate_indicator_parameters,
    normalize_data_for_trig,
    ensure_series_minimal_conversion,
    ensure_numpy_minimal_conversion,
)

# 公開API
__all__ = [
    # 新しいnumpy配列ベース指標クラス
    "TrendIndicators",
    "MomentumIndicators",
    "VolatilityIndicators",
    "VolumeIndicators",
    "PriceTransformIndicators",
    "StatisticsIndicators",
    "MathTransformIndicators",
    "MathOperatorsIndicators",
    "PatternRecognitionIndicators",
    "PandasTAError",
    "validate_input",
    "validate_series_data",
    "validate_indicator_parameters",
    "normalize_data_for_trig",
    "ensure_series_minimal_conversion",
    "ensure_numpy_minimal_conversion",
    # 既存クラス（互換性維持）
    "TechnicalIndicatorService",
]
