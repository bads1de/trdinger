"""
AutoStrategyConfigモジュール

オートストラテジーの全ての設定クラスを提供します。
"""

# 統合設定クラス
from .auto_strategy import (
    AutoStrategyConfig,
)

# 基底クラス
from .base import BaseConfig
from .ga import GASettings

# GA実行時設定クラス
from .ga_runtime import (
    GAConfig,
)
from .indicators import IndicatorSettings
from .position_sizing import PositionSizingSettings
from .tpsl import TPSLSettings

# 個別設定クラス
from .trading import TradingSettings

# __all__ で公開するクラスを定義
__all__ = [
    # 基底クラス
    "BaseConfig",
    # 個別設定クラス
    "TradingSettings",
    "IndicatorSettings",
    "GASettings",
    "TPSLSettings",
    "PositionSizingSettings",
    # 統合設定クラス
    "AutoStrategyConfig",
    # GA実行時設定クラス
    "GAConfig",
]


