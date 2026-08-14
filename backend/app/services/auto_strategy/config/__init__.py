"""
Auto Strategy Config モジュール

オートストラテジーの全ての設定クラスを提供します。

モジュール構成:
- constants.py: 共通定数・Enum・GA固有定数
- ga_config.py: GAConfig ランタイム設定 + ネスト設定（MutationConfig, EvaluationConfig 等）
- ga_validator.py: ConfigValidator バリデーション
- helpers.py: 設定ヘルパー関数（robustness regime window）
- evaluation_plan.py: robustness 評価計画
- indicator_universe.py: インジケーターユニバース定義と正規化
- objective_registry.py: 目的関数メタデータレジストリ
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# 目的関数レジストリ
from .evaluation_plan import (
    EvaluationPlan,
    RobustnessPlan,
)

# サブ設定クラス
from .ga_config import (
    EarlyTerminationSettings,
    EvaluationConfig,
    FitnessSharingConfig,
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
    from .ga_config import GAConfig
    from .ga_validator import ConfigValidator

_LAZY_EXPORTS = {
    "GAConfig": ".ga_config",
    "ConfigValidator": ".ga_validator",
}

__all__ = [
    "EarlyTerminationSettings",
    "MutationConfig",
    "EvaluationConfig",
    "TwoStageSelectionConfig",
    "RobustnessConfig",
    "IterativeImprovementConfig",
    "ValidationConfig",
    "FitnessSharingConfig",
    "EvaluationPlan",
    "RobustnessPlan",
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
