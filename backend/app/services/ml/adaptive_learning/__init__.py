"""
Adaptive Learning パッケージ

市場レジームの変化に対応してモデルを動的に調整・再学習します。
"""

from .adaptive_learning_service import (
    AdaptationResult,
    AdaptiveLearningConfig,
    AdaptiveLearningService,
)
from .market_regime_detector import (
    MarketRegime,
    MarketRegimeDetector,
    RegimeDetectionMethod,
    RegimeDetectionResult,
)

__all__ = [
    # Core classes
    "AdaptiveLearningService",
    "AdaptiveLearningConfig",
    "AdaptationResult",
    "MarketRegimeDetector",
    "MarketRegime",
    "RegimeDetectionMethod",
    "RegimeDetectionResult",
]
