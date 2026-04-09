"""
Auto Strategy Config モジュール

オートストラテジーの全ての設定クラスを提供します。
重い依存（pandas_ta_classic等）を避けるため、設定クラスは遅延インポートします。

モジュール構成:
- constants/: 共通定数・Enum・GA固有定数を統合
- ga/: GAConfig ランタイム設定（dataclass）、GAPresetsプリセット、ConfigValidatorバリデーション
- ga/nested_configs.py: GAConfig ネスト設定（MutationConfig, EvaluationConfig 等）
- auto_strategy_settings.py: AutoStrategyConfig 環境変数設定（pydantic）
- helpers/: 設定ヘルパー関数（ML filter/volatility gate、robustness regime window）
- indicator_universe.py: インジケーターユニバース定義と正規化
- objective_registry.py: 目的関数メタデータレジストリ
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# auto_strategy_settings.py は軽量（pydanticのみ）
from .auto_strategy_settings import AutoStrategyConfig

# サブ設定クラス
from .ga import (
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

# インジケーターユニバース
from .indicator_universe import (
    CURATED_INDICATOR_CATALOG,
    IndicatorUniverseMode,
    get_indicator_universe_names,
    is_in_indicator_universe,
    iter_indicator_universe_names,
    normalize_indicator_universe_mode,
    validate_indicator_universe,
)

# 目的関数レジストリ
from .objective_registry import (
    DEFAULT_OBJECTIVE_DEFINITION,
    OBJECTIVE_REGISTRY,
    ObjectiveDefinition,
    ObjectiveDirection,
    get_objective_definition,
    is_minimize_objective,
)

if TYPE_CHECKING:
    from .ga import ConfigValidator, GAConfig, GAPresets

_LAZY_EXPORTS = {
    "GAConfig": "GAConfig",
    "GAPresets": "GAPresets",
    "ConfigValidator": "ConfigValidator",
}


def __getattr__(name: str) -> Any:
    """遅延インポートで重い依存を回避"""
    attr_name = _LAZY_EXPORTS.get(name)
    if attr_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from . import ga

    value = getattr(ga, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *_LAZY_EXPORTS})


__all__ = [
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
    "GAConfig",
    "GAPresets",
    "ConfigValidator",
    "CURATED_INDICATOR_CATALOG",
    "IndicatorUniverseMode",
    "get_indicator_universe_names",
    "is_in_indicator_universe",
    "iter_indicator_universe_names",
    "normalize_indicator_universe_mode",
    "validate_indicator_universe",
    "DEFAULT_OBJECTIVE_DEFINITION",
    "OBJECTIVE_REGISTRY",
    "ObjectiveDefinition",
    "ObjectiveDirection",
    "get_objective_definition",
    "is_minimize_objective",
]
