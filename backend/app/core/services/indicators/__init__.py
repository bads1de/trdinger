"""
テクニカル指標パッケージ

分割されたテクニカル指標クラスと統合サービスを提供します。
新しいnumpy配列ベースのオートストラテジー最適化版も含みます。
"""

from typing import Dict, Any

# 新しいnumpy配列ベース指標クラス（オートストラテジー最適化版）
from .trend import TrendIndicators
from .momentum import MomentumIndicators
from .volatility import VolatilityIndicators
from .utils import TALibError, validate_input, ensure_numpy_array

# 既存のクラス（互換性維持）
try:
    from .indicator_orchestrator import TechnicalIndicatorService
    from .abstract_indicator import BaseIndicator
except ImportError:
    # 依存関係が不足している場合はスキップ
    TechnicalIndicatorService = None
    BaseIndicator = None

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
    "BaseIndicator",
]
