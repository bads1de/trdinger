"""
TA-Lib アダプターモジュール

機能別に分割されたTA-Libアダプタークラスを提供します。
各アダプターは特定のカテゴリのテクニカル指標を担当し、
単一責任原則に従った設計となっています。
"""

from .base_adapter import BaseAdapter, TALibCalculationError
from .trend_adapter import TrendAdapter
from .momentum_adapter import MomentumAdapter
from .volatility_adapter import VolatilityAdapter
from .volume_adapter import VolumeAdapter
from .price_transform_adapter import PriceTransformAdapter

__all__ = [
    "BaseAdapter",
    "TALibCalculationError",
    "TrendAdapter",
    "MomentumAdapter",
    "VolatilityAdapter",
    "VolumeAdapter",
    "PriceTransformAdapter",
]
