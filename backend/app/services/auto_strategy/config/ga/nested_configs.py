"""
GAConfig ネスト設定

GAConfig にぶら下がる設定 dataclass 群と legacy key 変換ルールを定義する。
GAConfig のフラットフィールドと併存し、ネスト辞書からの復元にも対応する。
"""

import logging
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, List, Mapping, Optional

from ..constants import GA_DEFAULT_CONFIG, GA_MUTATION_SETTINGS, OPERATORS

logger = logging.getLogger(__name__)


EARLY_TERMINATION_FLAT_FIELD_MAP = {
    "enabled": "enable_early_termination",
    "max_drawdown": "early_termination_max_drawdown",
    "min_trades": "early_termination_min_trades",
    "min_trade_check_progress": "early_termination_min_trade_check_progress",
    "trade_pace_tolerance": "early_termination_trade_pace_tolerance",
    "min_expectancy": "early_termination_min_expectancy",
    "expectancy_min_trades": "early_termination_expectancy_min_trades",
    "expectancy_progress": "early_termination_expectancy_progress",
}
EARLY_TERMINATION_LEGACY_FIELD_MAP = {
    legacy_key: field_name
    for field_name, legacy_key in EARLY_TERMINATION_FLAT_FIELD_MAP.items()
}

MUTATION_FLAT_FIELD_MAP = {
    "rate": "mutation_rate",
    "crossover_field_selection_probability": "crossover_field_selection_probability",
    "indicator_param_range": "indicator_param_mutation_range",
    "risk_param_range": "risk_param_mutation_range",
    "indicator_add_delete_probability": "indicator_add_delete_probability",
    "indicator_add_vs_delete_probability": "indicator_add_vs_delete_probability",
    "condition_change_multiplier": "condition_change_probability_multiplier",
    "condition_selection_probability": "condition_selection_probability",
    "condition_operator_switch_probability": "condition_operator_switch_probability",
    "tpsl_gene_creation_multiplier": "tpsl_gene_creation_probability_multiplier",
    "position_sizing_gene_creation_multiplier": "position_sizing_gene_creation_probability_multiplier",
    "adaptive_variance_threshold": "adaptive_mutation_variance_threshold",
    "adaptive_decrease_multiplier": "adaptive_mutation_rate_decrease_multiplier",
    "adaptive_increase_multiplier": "adaptive_mutation_rate_increase_multiplier",
    "valid_condition_operators": "valid_condition_operators",
}
MUTATION_LEGACY_FIELD_MAP = {
    legacy_key: field_name for field_name, legacy_key in MUTATION_FLAT_FIELD_MAP.items()
}

EVALUATION_FLAT_FIELD_MAP = {
    "enable_parallel": "enable_parallel_evaluation",
    "max_workers": "max_evaluation_workers",
    "timeout": "evaluation_timeout",
    "enable_multi_fidelity_evaluation": "enable_multi_fidelity_evaluation",
    "multi_fidelity_window_ratio": "multi_fidelity_window_ratio",
    "multi_fidelity_oos_ratio": "multi_fidelity_oos_ratio",
    "multi_fidelity_candidate_ratio": "multi_fidelity_candidate_ratio",
    "multi_fidelity_min_candidates": "multi_fidelity_min_candidates",
    "oos_split_ratio": "oos_split_ratio",
    "oos_fitness_weight": "oos_fitness_weight",
    "enable_walk_forward": "enable_walk_forward",
    "wfa_n_folds": "wfa_n_folds",
    "wfa_train_ratio": "wfa_train_ratio",
    "wfa_anchored": "wfa_anchored",
}
EVALUATION_LEGACY_FIELD_MAP = {
    legacy_key: field_name
    for field_name, legacy_key in EVALUATION_FLAT_FIELD_MAP.items()
}

HYBRID_FLAT_FIELD_MAP = {
    "mode": "hybrid_mode",
    "model_type": "hybrid_model_type",
    "model_types": "hybrid_model_types",
    "volatility_gate_enabled": "volatility_gate_enabled",
    "volatility_model_path": "volatility_model_path",
    "ml_filter_enabled": "ml_filter_enabled",
    "ml_model_path": "ml_model_path",
    "preprocess_features": "preprocess_features",
}
HYBRID_LEGACY_FIELD_MAP = {
    legacy_key: field_name for field_name, legacy_key in HYBRID_FLAT_FIELD_MAP.items()
}

TUNING_FLAT_FIELD_MAP = {
    "enabled": "enable_parameter_tuning",
    "n_trials": "tuning_n_trials",
    "elite_count": "tuning_elite_count",
    "use_wfa": "tuning_use_wfa",
    "include_indicators": "tuning_include_indicators",
    "include_tpsl": "tuning_include_tpsl",
    "include_thresholds": "tuning_include_thresholds",
}
TUNING_LEGACY_FIELD_MAP = {
    legacy_key: field_name for field_name, legacy_key in TUNING_FLAT_FIELD_MAP.items()
}

TWO_STAGE_SELECTION_FLAT_FIELD_MAP = {
    "enabled": "enable_two_stage_selection",
    "elite_count": "two_stage_elite_count",
    "candidate_pool_size": "two_stage_candidate_pool_size",
    "min_pass_rate": "two_stage_min_pass_rate",
}
TWO_STAGE_SELECTION_LEGACY_FIELD_MAP = {
    legacy_key: field_name
    for field_name, legacy_key in TWO_STAGE_SELECTION_FLAT_FIELD_MAP.items()
}

ROBUSTNESS_FLAT_FIELD_MAP = {
    "validation_symbols": "robustness_validation_symbols",
    "regime_windows": "robustness_regime_windows",
    "stress_slippage": "robustness_stress_slippage",
    "stress_commission_multipliers": "robustness_stress_commission_multipliers",
    "aggregate_method": "robustness_aggregate_method",
}
ROBUSTNESS_LEGACY_FIELD_MAP = {
    legacy_key: field_name
    for field_name, legacy_key in ROBUSTNESS_FLAT_FIELD_MAP.items()
}

DEFAULT_EARLY_TERMINATION_VALUES = {
    "enabled": False,
    "max_drawdown": None,
    "min_trades": None,
    "min_trade_check_progress": 0.5,
    "trade_pace_tolerance": 0.5,
    "min_expectancy": None,
    "expectancy_min_trades": 5,
    "expectancy_progress": 0.6,
}


def _read_value(source: Any, key: str, default: Any = None) -> Any:
    """dict / オブジェクトのどちらからでも値を取得する。"""
    if isinstance(source, Mapping):
        if key in source:
            return source[key]
        return default

    try:
        values = vars(source)
    except TypeError:
        values = None

    if values is not None:
        if key in values:
            return values[key]
        return default

    return getattr(source, key, default)


def _filter_known_fields(
    cls, data: Mapping[str, Any], config_name: str
) -> dict[str, Any]:
    """既知フィールドのみを残し、未知キーは警告する。"""
    known = {field_info.name for field_info in fields(cls)}
    unknown = sorted(key for key in data.keys() if key not in known)
    if unknown:
        logger.warning(
            "%s の未対応キーを無視しました: %s",
            config_name,
            ", ".join(unknown),
        )
    return {key: value for key, value in data.items() if key in known}


def _clone_flat_value(value: Any) -> Any:
    """flat dict へ出す値を安全に複製する。"""
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def _read_value_or_default(
    source: Any,
    key: str,
    default: Any,
    coercer: Optional[type] = None,
) -> Any:
    """値を読み取り、None は既定値へ寄せた上で必要なら型変換する。"""
    value = _read_value(source, key, default)
    if value is None:
        value = default
    return coercer(value) if coercer is not None else value


class NestedConfigMixin:
    """ネスト設定 dataclass の共通変換処理。"""

    FLAT_FIELD_MAP: ClassVar[dict[str, str]] = {}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        filtered = _filter_known_fields(cls, dict(data), cls.__name__)
        return cls(**filtered)

    def to_flat_dict(self) -> dict[str, Any]:
        return {
            legacy_key: _clone_flat_value(getattr(self, field_name))
            for field_name, legacy_key in self.FLAT_FIELD_MAP.items()
        }


@dataclass(frozen=True)
class EarlyTerminationSettings(NestedConfigMixin):
    """早期終了の正規化済み設定。"""

    FLAT_FIELD_MAP: ClassVar[dict[str, str]] = EARLY_TERMINATION_FLAT_FIELD_MAP

    enabled: bool = False
    max_drawdown: Optional[float] = None
    min_trades: Optional[int] = None
    min_trade_check_progress: float = 0.5
    trade_pace_tolerance: float = 0.5
    min_expectancy: Optional[float] = None
    expectancy_min_trades: int = 5
    expectancy_progress: float = 0.6

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """設定フィールド名を返す。"""
        return tuple(field_info.name for field_info in fields(cls))

    @classmethod
    def from_source(cls, source: Any) -> "EarlyTerminationSettings":
        """dict / オブジェクトのどちらからでも設定を生成する。"""
        if isinstance(source, cls):
            return source
        if isinstance(source, Mapping):
            return cls.from_dict(source)

        filtered = {
            field_name: _read_value(source, field_name, default)
            for field_name, default in DEFAULT_EARLY_TERMINATION_VALUES.items()
        }
        return cls(**filtered)

    @classmethod
    def from_legacy_source(cls, source: Any) -> "EarlyTerminationSettings":
        """フラットな legacy フィールドから設定を生成する。"""
        return cls(
            enabled=bool(
                _read_value_or_default(
                    source,
                    EARLY_TERMINATION_FLAT_FIELD_MAP["enabled"],
                    DEFAULT_EARLY_TERMINATION_VALUES["enabled"],
                )
            ),
            max_drawdown=_read_value(source, "early_termination_max_drawdown", None),
            min_trades=_read_value(source, "early_termination_min_trades", None),
            min_trade_check_progress=_read_value_or_default(
                source,
                EARLY_TERMINATION_FLAT_FIELD_MAP["min_trade_check_progress"],
                DEFAULT_EARLY_TERMINATION_VALUES["min_trade_check_progress"],
                float,
            ),
            trade_pace_tolerance=_read_value_or_default(
                source,
                EARLY_TERMINATION_FLAT_FIELD_MAP["trade_pace_tolerance"],
                DEFAULT_EARLY_TERMINATION_VALUES["trade_pace_tolerance"],
                float,
            ),
            min_expectancy=_read_value(
                source, "early_termination_min_expectancy", None
            ),
            expectancy_min_trades=_read_value_or_default(
                source,
                EARLY_TERMINATION_FLAT_FIELD_MAP["expectancy_min_trades"],
                DEFAULT_EARLY_TERMINATION_VALUES["expectancy_min_trades"],
                int,
            ),
            expectancy_progress=_read_value_or_default(
                source,
                EARLY_TERMINATION_FLAT_FIELD_MAP["expectancy_progress"],
                DEFAULT_EARLY_TERMINATION_VALUES["expectancy_progress"],
                float,
            ),
        )

    def to_strategy_params(self) -> dict[str, Any]:
        """strategy params 互換の legacy フィールドに変換する。"""
        return self.to_flat_dict()


def _coerce_early_termination_settings(
    value: Any,
) -> Optional[EarlyTerminationSettings]:
    """nested 値が設定オブジェクトとして妥当な場合のみ正規化する。"""
    if value is None:
        return None
    if isinstance(value, EarlyTerminationSettings):
        return value
    if isinstance(value, Mapping):
        return EarlyTerminationSettings.from_source(value)

    try:
        values = vars(value)
    except TypeError:
        return None

    if not any(name in values for name in EarlyTerminationSettings.field_names()):
        return None

    return EarlyTerminationSettings.from_source(value)


def resolve_early_termination_settings(source: Any) -> EarlyTerminationSettings:
    """nested / legacy flat のいずれからでも早期終了設定を解決する。"""
    if isinstance(source, EarlyTerminationSettings):
        return source

    nested = _coerce_early_termination_settings(
        _read_value(source, "early_termination_settings", None)
    )
    if nested is not None:
        return nested
    return EarlyTerminationSettings.from_legacy_source(source)


@dataclass
class MutationConfig(NestedConfigMixin):
    """突然変異関連設定。"""

    FLAT_FIELD_MAP: ClassVar[dict[str, str]] = MUTATION_FLAT_FIELD_MAP

    rate: float = float(GA_DEFAULT_CONFIG["mutation_rate"])
    crossover_field_selection_probability: float = GA_MUTATION_SETTINGS[
        "crossover_field_selection_probability"
    ]
    indicator_param_range: List[float] = field(
        default_factory=lambda: list(
            GA_MUTATION_SETTINGS["indicator_param_mutation_range"]
        )
    )
    risk_param_range: List[float] = field(
        default_factory=lambda: list(GA_MUTATION_SETTINGS["risk_param_mutation_range"])
    )
    indicator_add_delete_probability: float = GA_MUTATION_SETTINGS[
        "indicator_add_delete_probability"
    ]
    indicator_add_vs_delete_probability: float = GA_MUTATION_SETTINGS[
        "indicator_add_vs_delete_probability"
    ]
    condition_change_multiplier: float = GA_MUTATION_SETTINGS[
        "condition_change_probability_multiplier"
    ]
    condition_selection_probability: float = GA_MUTATION_SETTINGS[
        "condition_selection_probability"
    ]
    condition_operator_switch_probability: float = GA_MUTATION_SETTINGS[
        "condition_operator_switch_probability"
    ]
    tpsl_gene_creation_multiplier: float = GA_MUTATION_SETTINGS[
        "tpsl_gene_creation_probability_multiplier"
    ]
    position_sizing_gene_creation_multiplier: float = GA_MUTATION_SETTINGS[
        "position_sizing_gene_creation_probability_multiplier"
    ]
    adaptive_variance_threshold: float = GA_MUTATION_SETTINGS[
        "adaptive_mutation_variance_threshold"
    ]
    adaptive_decrease_multiplier: float = GA_MUTATION_SETTINGS[
        "adaptive_mutation_rate_decrease_multiplier"
    ]
    adaptive_increase_multiplier: float = GA_MUTATION_SETTINGS[
        "adaptive_mutation_rate_increase_multiplier"
    ]
    valid_condition_operators: List[str] = field(
        default_factory=lambda: OPERATORS.copy()
    )


@dataclass
class EvaluationConfig(NestedConfigMixin):
    """評価・検証関連設定。"""

    FLAT_FIELD_MAP: ClassVar[dict[str, str]] = EVALUATION_FLAT_FIELD_MAP

    enable_parallel: bool = True
    max_workers: Optional[int] = None
    timeout: float = 300.0
    enable_multi_fidelity_evaluation: bool = False
    multi_fidelity_window_ratio: float = 0.3
    multi_fidelity_oos_ratio: float = 0.2
    multi_fidelity_candidate_ratio: float = 0.25
    multi_fidelity_min_candidates: int = 3
    early_termination_settings: EarlyTerminationSettings = field(
        default_factory=EarlyTerminationSettings
    )
    oos_split_ratio: float = 0.0
    oos_fitness_weight: float = 0.5
    enable_walk_forward: bool = False
    wfa_n_folds: int = 5
    wfa_train_ratio: float = 0.7
    wfa_anchored: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationConfig":
        """
        辞書からEvaluationConfigインスタンスを生成

        Args:
            data: 設定値を含む辞書。未知のキーは無視されます。

        Returns:
            初期化されたEvaluationConfigインスタンス
        """
        working = dict(data)
        legacy_payload = {}
        for legacy_key, canonical_key in EARLY_TERMINATION_LEGACY_FIELD_MAP.items():
            if legacy_key not in working:
                continue
            value = working.pop(legacy_key)
            if "early_termination_settings" not in working and value is not None:
                legacy_payload[canonical_key] = value
        if legacy_payload:
            working["early_termination_settings"] = legacy_payload

        filtered = _filter_known_fields(cls, working, "EvaluationConfig")
        settings = filtered.get("early_termination_settings")
        if settings is None:
            filtered["early_termination_settings"] = EarlyTerminationSettings()
        elif not isinstance(settings, EarlyTerminationSettings):
            filtered["early_termination_settings"] = (
                EarlyTerminationSettings.from_source(settings)
            )
        return cls(**filtered)

    def to_flat_dict(self) -> dict[str, Any]:
        """GAConfig のフラット設定名へ変換する。"""
        flat = super().to_flat_dict()
        flat.update(self.early_termination_settings.to_strategy_params())
        return flat


@dataclass
class HybridConfig(NestedConfigMixin):
    """ハイブリッドGA+ML関連設定。"""

    FLAT_FIELD_MAP: ClassVar[dict[str, str]] = HYBRID_FLAT_FIELD_MAP

    mode: bool = False
    model_type: str = "lightgbm"
    model_types: Optional[List[str]] = None
    volatility_gate_enabled: bool = False
    volatility_model_path: Optional[str] = None
    ml_filter_enabled: bool = False
    ml_model_path: Optional[str] = None
    preprocess_features: bool = True


@dataclass
class TuningConfig(NestedConfigMixin):
    """パラメータチューニング（Optuna）関連設定。"""

    FLAT_FIELD_MAP: ClassVar[dict[str, str]] = TUNING_FLAT_FIELD_MAP

    enabled: bool = True
    n_trials: int = 30
    elite_count: int = 3
    use_wfa: bool = True
    include_indicators: bool = True
    include_tpsl: bool = True
    include_thresholds: bool = False


@dataclass
class TwoStageSelectionConfig(NestedConfigMixin):
    """二段階選抜関連設定。"""

    FLAT_FIELD_MAP: ClassVar[dict[str, str]] = TWO_STAGE_SELECTION_FLAT_FIELD_MAP

    enabled: bool = True
    elite_count: int = 3
    candidate_pool_size: int = 5
    min_pass_rate: float = 0.5


@dataclass
class RobustnessConfig(NestedConfigMixin):
    """robustness 評価関連設定。"""

    FLAT_FIELD_MAP: ClassVar[dict[str, str]] = ROBUSTNESS_FLAT_FIELD_MAP

    validation_symbols: Optional[List[str]] = None
    regime_windows: List[dict] = field(default_factory=list)
    stress_slippage: List[float] = field(default_factory=list)
    stress_commission_multipliers: List[float] = field(default_factory=list)
    aggregate_method: str = "robust"


__all__ = [
    "EARLY_TERMINATION_FLAT_FIELD_MAP",
    "EARLY_TERMINATION_LEGACY_FIELD_MAP",
    "MUTATION_FLAT_FIELD_MAP",
    "MUTATION_LEGACY_FIELD_MAP",
    "EVALUATION_FLAT_FIELD_MAP",
    "EVALUATION_LEGACY_FIELD_MAP",
    "HYBRID_FLAT_FIELD_MAP",
    "HYBRID_LEGACY_FIELD_MAP",
    "TUNING_FLAT_FIELD_MAP",
    "TUNING_LEGACY_FIELD_MAP",
    "TWO_STAGE_SELECTION_FLAT_FIELD_MAP",
    "TWO_STAGE_SELECTION_LEGACY_FIELD_MAP",
    "ROBUSTNESS_FLAT_FIELD_MAP",
    "ROBUSTNESS_LEGACY_FIELD_MAP",
    "EarlyTerminationSettings",
    "resolve_early_termination_settings",
    "MutationConfig",
    "EvaluationConfig",
    "HybridConfig",
    "TuningConfig",
    "TwoStageSelectionConfig",
    "RobustnessConfig",
]
