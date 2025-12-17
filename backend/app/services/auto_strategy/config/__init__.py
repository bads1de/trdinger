"""
Auto Strategy Config モジュール

オートストラテジーの全ての設定クラスを提供します。
"""

# 基底クラス
from .base import BaseConfig

# GA実行時設定クラス（メイン）
from .ga import GAConfig

# 個別設定クラス（統合済み）
from .settings import (
    IndicatorSettings,
    PositionSizingSettings,
    TPSLSettings,
    TradingSettings,
)

# 後方互換性のためのエイリアス
GASettings = GAConfig

# __all__ で公開するクラスを定義
__all__ = [
    # 基底クラス
    "BaseConfig",
    # 個別設定クラス
    "TradingSettings",
    "IndicatorSettings",
    "GASettings",  # 後方互換性（GAConfigのエイリアス）
    "GAConfig",
    "TPSLSettings",
    "PositionSizingSettings",
]
