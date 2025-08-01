"""
テクニカル指標パッケージ

分割されたテクニカル指標クラスと統合サービスを提供します。
新しいnumpy配列ベースのオートストラテジー最適化版も含みます。
"""

from typing import Dict, Any

# 新しいnumpy配列ベース指標クラス（オートストラテジー最適化版）
from .technical_indicators.trend import TrendIndicators
from .technical_indicators.momentum import MomentumIndicators
from .technical_indicators.volatility import VolatilityIndicators
from .technical_indicators.volume import VolumeIndicators
from .technical_indicators.price_transform import PriceTransformIndicators
from .technical_indicators.cycle import CycleIndicators
from .technical_indicators.statistics import StatisticsIndicators
from .technical_indicators.math_transform import MathTransformIndicators
from .technical_indicators.math_operators import MathOperatorsIndicators
from .technical_indicators.pattern_recognition import PatternRecognitionIndicators
from .utils import TALibError, validate_input, ensure_numpy_array

# 既存のクラス（互換性維持）
from .indicator_orchestrator import TechnicalIndicatorService




# 公開API
__all__ = [
    # 新しいnumpy配列ベース指標クラス（オートストラテジー最適化版）
    "TrendIndicators",
    "MomentumIndicators",
    "VolatilityIndicators",
    "VolumeIndicators",
    "PriceTransformIndicators",
    "CycleIndicators",
    "StatisticsIndicators",
    "MathTransformIndicators",
    "MathOperatorsIndicators",
    "PatternRecognitionIndicators",
    "TALibError",
    "validate_input",
    "ensure_numpy_array",
    # 既存クラス（互換性維持）
    "TechnicalIndicatorService",
]
