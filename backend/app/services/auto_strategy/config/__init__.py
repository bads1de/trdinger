"""
AutoStrategyConfigモジュール

オートストラテジーの全ての設定クラスを提供します。
"""

# 基底クラス
from .base import BaseConfig

# 個別設定クラス
from .trading import TradingSettings
from .indicators import IndicatorSettings
from .ga import GASettings
from .tpsl import TPSLSettings
from .position_sizing import PositionSizingSettings

# 統合設定クラス
from .auto_strategy import (
    AutoStrategyConfig,
    DEFAULT_AUTO_STRATEGY_CONFIG,
    get_default_config,
    create_config_from_file,
    validate_config_file,
)

# GA実行時設定クラス
from .ga_runtime import (
    GAConfig,
    GAProgress,
)

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
    "DEFAULT_AUTO_STRATEGY_CONFIG",
    "get_default_config",
    "create_config_from_file",
    "validate_config_file",
    # GA実行時設定クラス
    "GAConfig",
    "GAProgress",
]