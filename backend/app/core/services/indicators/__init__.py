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
from .utils import TALibError, validate_input, ensure_numpy_array

# 既存のクラス（互換性維持）
from .indicator_orchestrator import TechnicalIndicatorService

# 古いインポートは一時的にコメントアウト（段階的移行のため）
# from .momentum_indicators import ...
# from .volatility_indicators import ...
# from .volume_indicators import ...
# from .price_transform_indicators import ...


# 公開API
__all__ = [
    # 新しいnumpy配列ベース指標クラス（オートストラテジー最適化版）
    "TrendIndicators",
    "MomentumIndicators",
    "VolatilityIndicators",
    "TALibError",
    "validate_input",
    "ensure_numpy_array",
    # 既存クラス（互換性維持）
    "TechnicalIndicatorService",
]
