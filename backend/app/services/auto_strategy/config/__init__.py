"""
Auto Strategy Config モジュール

オートストラテジーの全ての設定クラスを提供します。
重い依存（pandas_ta_classic等）を避けるため、設定クラスは遅延インポートします。

モジュール構成:
- constants/: 共通定数・Enum・GA固有定数を統合
- ga/: GAConfig ランタイム設定（dataclass）、GAPresetsプリセット、ConfigValidatorバリデーション
- ga/nested_configs.py: GAConfig ネスト設定（MutationConfig, EvaluationConfig 等）
- helpers/: 設定ヘルパー関数（volatility gate、robustness regime window）
- indicator_universe.py: インジケーターユニバース定義と正規化
- objective_registry.py: 目的関数メタデータレジストリ
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# 目的関数レジストリ
from .evaluation_plan import (
    DateRange,
    EvaluationPlan,
    ExecutionPlan,
    FinalTestPlan,
    RobustnessPlan,
    SelectionPlan,
    ValidationPlan,
)

# サブ設定クラス
from .ga import (
    EarlyTerminationSettings,
    EvaluationConfig,
    IterativeImprovementConfig,
    MutationConfig,
    RobustnessConfig,
    TwoStageSelectionConfig,
    ValidationConfig,
)

# ヘルパー関数（軽量、依存なし）
from .helpers import (
    RobustnessRegimeWindow,
    normalize_robustness_regime_window,
    normalize_robustness_regime_windows,
    validate_robustness_regime_window,
)

# インジケーターユニバース
from .indicator_universe import (
    CURATED_INDICATOR_CATALOG,
    IndicatorUniverseMode,
    get_indicator_universe_names,
    is_indicator_in_universe,
    iter_indicator_universe_names,
    normalize_indicator_universe_mode,
)
from .objective_registry import (
    DEFAULT_OBJECTIVE_DEFINITION,
    OBJECTIVE_REGISTRY,
    ObjectiveDefinition,
    ObjectiveDirection,
    get_objective_definition,
    is_minimize_objective,
)

if TYPE_CHECKING:
    from .ga import ConfigValidator, GAConfig

_LAZY_EXPORTS = {
    "GAConfig": ".ga",
    "ConfigValidator": ".ga",
}

__all__ = [
    "EarlyTerminationSettings",
    "MutationConfig",
    "EvaluationConfig",
    "TwoStageSelectionConfig",
    "RobustnessConfig",
    "IterativeImprovementConfig",
    "ValidationConfig",
    "DateRange",
    "EvaluationPlan",
    "ExecutionPlan",
    "FinalTestPlan",
    "RobustnessPlan",
    "SelectionPlan",
    "ValidationPlan",
    "RobustnessRegimeWindow",
    "normalize_robustness_regime_window",
    "normalize_robustness_regime_windows",
    "validate_robustness_regime_window",
    "GAConfig",
    "ConfigValidator",
    "CURATED_INDICATOR_CATALOG",
    "IndicatorUniverseMode",
    "get_indicator_universe_names",
    "is_indicator_in_universe",
    "iter_indicator_universe_names",
    "normalize_indicator_universe_mode",
    "DEFAULT_OBJECTIVE_DEFINITION",
    "OBJECTIVE_REGISTRY",
    "ObjectiveDefinition",
    "ObjectiveDirection",
    "get_objective_definition",
    "is_minimize_objective",
]

from app.utils.lazy_import import setup_lazy_import  # noqa: E402

setup_lazy_import(globals(), _LAZY_EXPORTS)
