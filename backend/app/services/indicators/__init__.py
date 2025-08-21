"""
テクニカル指標パッケージ

分割されたテクニカル指標クラスと統合サービスを提供します。
新しいnumpy配列ベースのオートストラテジー最適化版も含みます。
"""

from .indicator_orchestrator import TechnicalIndicatorService

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
)

# 公開API
__all__ = [
    "TrendIndicators",
    "MomentumIndicators",
    "VolatilityIndicators",
    "VolumeIndicators",
    "PriceTransformIndicators",
    "StatisticsIndicators",
    "PatternRecognitionIndicators",
    "PandasTAError",
    "validate_input",
    "TechnicalIndicatorService",
]
