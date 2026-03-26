"""
Auto Strategy Config モジュール

オートストラテジーの全ての設定クラスを提供します。
重い依存（pandas_ta_classic等）を避けるため、設定クラスは遅延インポートします。

モジュール構成:
- base.py: BaseConfig 基底クラス
- constants.py: 共通定数・Enum・GA固有定数を統合
- ga.py: GAConfig ランタイム設定（dataclass）
- auto_strategy_settings.py: AutoStrategyConfig 環境変数設定（pydantic）
- trading_settings.py: TradingSettings 取引基本設定
- indicator_settings.py: IndicatorSettings テクニカル指標設定
- tpsl_settings.py: TPSLSettings TP/SL設定
- position_sizing_settings.py: PositionSizingSettings ポジションサイジング設定
- validators.py: ConfigValidator バリデーション
"""

# 基底クラス（軽量、依存なし）
from .base import BaseConfig

# auto_strategy_settings.py は軽量（pydanticのみ）
from .auto_strategy_settings import AutoStrategyConfig


def __getattr__(name: str):
    """遅延インポートで重い依存を回避"""
    if name == "GAConfig":
        from .ga import GAConfig
        return GAConfig
    if name in ("IndicatorSettings", "PositionSizingSettings", "TPSLSettings", "TradingSettings"):
        from .settings import (
            IndicatorSettings,
            PositionSizingSettings,
            TPSLSettings,
            TradingSettings,
        )
        return {
            "IndicatorSettings": IndicatorSettings,
            "PositionSizingSettings": PositionSizingSettings,
            "TPSLSettings": TPSLSettings,
            "TradingSettings": TradingSettings,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseConfig",
    "AutoStrategyConfig",
    "GAConfig",
    "TradingSettings",
    "IndicatorSettings",
    "TPSLSettings",
    "PositionSizingSettings",
]
