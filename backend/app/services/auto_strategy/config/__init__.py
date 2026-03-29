"""
Auto Strategy Config モジュール

オートストラテジーの全ての設定クラスを提供します。
重い依存（pandas_ta_classic等）を避けるため、設定クラスは遅延インポートします。

モジュール構成:
- base.py: BaseConfig 基底クラス
- constants.py: 共通定数・Enum・GA固有定数を統合
- ga.py: GAConfig ランタイム設定（dataclass）
- sub_configs.py: GAConfig サブ設定（MutationConfig, EvaluationConfig 等）
- auto_strategy_settings.py: AutoStrategyConfig 環境変数設定（pydantic）
- validators.py: ConfigValidator バリデーション
"""

# 基底クラス（軽量、依存なし）
from .base import BaseConfig

# auto_strategy_settings.py は軽量（pydanticのみ）
from .auto_strategy_settings import AutoStrategyConfig

# サブ設定クラス
from .sub_configs import (
    EvaluationConfig,
    HybridConfig,
    MutationConfig,
    RobustnessConfig,
    TuningConfig,
    TwoStageSelectionConfig,
)


def __getattr__(name: str):
    """遅延インポートで重い依存を回避"""
    if name == "GAConfig":
        from .ga import GAConfig

        return GAConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseConfig",
    "AutoStrategyConfig",
    "GAConfig",
    "MutationConfig",
    "EvaluationConfig",
    "HybridConfig",
    "TuningConfig",
    "TwoStageSelectionConfig",
    "RobustnessConfig",
]
