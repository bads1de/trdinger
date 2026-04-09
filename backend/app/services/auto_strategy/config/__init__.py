"""
Auto Strategy Config モジュール

オートストラテジーの全ての設定クラスを提供します。
重い依存（pandas_ta_classic等）を避けるため、設定クラスは遅延インポートします。

モジュール構成:
- base.py: BaseConfig 基底クラス
- constants.py: 共通定数・Enum・GA固有定数を統合
- ga.py: GAConfig ランタイム設定（dataclass）、GAPresetsプリセット、ConfigValidatorバリデーション
- ga_nested_configs.py: GAConfig ネスト設定（MutationConfig, EvaluationConfig 等）
- auto_strategy_settings.py: AutoStrategyConfig 環境変数設定（pydantic）
- helpers.py: 設定ヘルパー関数（ML filter/volatility gate、robustness regime window）
"""

# auto_strategy_settings.py は軽量（pydanticのみ）
from .auto_strategy_settings import AutoStrategyConfig

# 基底クラス（軽量、依存なし）
from .base import BaseConfig

# サブ設定クラス
from .ga_nested_configs import (
    EarlyTerminationSettings,
    EvaluationConfig,
    HybridConfig,
    MutationConfig,
    RobustnessConfig,
    TuningConfig,
    TwoStageSelectionConfig,
)

# ヘルパー関数（軽量、依存なし）
from .helpers import (
    MLGateSettings,
    RobustnessRegimeWindow,
    normalize_ml_gate_fields,
    normalize_robustness_regime_window,
    normalize_robustness_regime_windows,
    resolve_ml_gate_settings,
    validate_robustness_regime_window,
)


def __getattr__(name: str):
    """遅延インポートで重い依存を回避"""
    if name == "GAConfig":
        from .ga import GAConfig

        return GAConfig
    if name == "GAPresets":
        from .ga import GAPresets

        return GAPresets
    if name == "ConfigValidator":
        from .ga import ConfigValidator

        return ConfigValidator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseConfig",
    "AutoStrategyConfig",
    "EarlyTerminationSettings",
    "MutationConfig",
    "EvaluationConfig",
    "HybridConfig",
    "TuningConfig",
    "TwoStageSelectionConfig",
    "RobustnessConfig",
    "MLGateSettings",
    "RobustnessRegimeWindow",
    "resolve_ml_gate_settings",
    "normalize_ml_gate_fields",
    "normalize_robustness_regime_window",
    "normalize_robustness_regime_windows",
    "validate_robustness_regime_window",
]
