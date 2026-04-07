"""
GA設定サブクラス

GAConfig の設定項目を意味的にグループ化するサブデータクラス群。
GAConfig のフラットフィールドと併存し、ネスト辞書からの復元にも対応する。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional

logger = logging.getLogger(__name__)


EARLY_TERMINATION_LEGACY_FIELD_MAP = {
    "enable_early_termination": "enabled",
    "early_termination_max_drawdown": "max_drawdown",
    "early_termination_min_trades": "min_trades",
    "early_termination_min_trade_check_progress": "min_trade_check_progress",
    "early_termination_trade_pace_tolerance": "trade_pace_tolerance",
    "early_termination_min_expectancy": "min_expectancy",
    "early_termination_expectancy_min_trades": "expectancy_min_trades",
    "early_termination_expectancy_progress": "expectancy_progress",
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


def _filter_known_fields(cls, data: dict, config_name: str) -> dict:
    """既知フィールドのみを残し、未知キーは警告する。"""
    known = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = sorted(key for key in data.keys() if key not in known)
    if unknown:
        logger.warning(
            "%s の未対応キーを無視しました: %s",
            config_name,
            ", ".join(unknown),
        )
    return {key: value for key, value in data.items() if key in known}


@dataclass(frozen=True)
class EarlyTerminationSettings:
    """早期終了の正規化済み設定。"""

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
        return tuple(cls.__dataclass_fields__.keys())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EarlyTerminationSettings":
        """辞書から EarlyTerminationSettings インスタンスを生成する。"""
        filtered = _filter_known_fields(cls, dict(data), "EarlyTerminationSettings")
        return cls(**filtered)

    @classmethod
    def from_source(cls, source: Any) -> "EarlyTerminationSettings":
        """dict / オブジェクトのどちらからでも設定を生成する。"""
        if isinstance(source, cls):
            return source
        if isinstance(source, Mapping):
            return cls.from_dict(source)

        filtered = {
            field_name: _read_value(source, field_name, default)
            for field_name, default in {
                "enabled": False,
                "max_drawdown": None,
                "min_trades": None,
                "min_trade_check_progress": 0.5,
                "trade_pace_tolerance": 0.5,
                "min_expectancy": None,
                "expectancy_min_trades": 5,
                "expectancy_progress": 0.6,
            }.items()
        }
        return cls(**filtered)

    @classmethod
    def from_legacy_source(cls, source: Any) -> "EarlyTerminationSettings":
        """フラットな legacy フィールドから設定を生成する。"""
        min_trade_check_progress = _read_value(
            source, "early_termination_min_trade_check_progress", 0.5
        )
        if min_trade_check_progress is None:
            min_trade_check_progress = 0.5

        trade_pace_tolerance = _read_value(
            source, "early_termination_trade_pace_tolerance", 0.5
        )
        if trade_pace_tolerance is None:
            trade_pace_tolerance = 0.5

        expectancy_min_trades = _read_value(
            source, "early_termination_expectancy_min_trades", 5
        )
        if expectancy_min_trades is None:
            expectancy_min_trades = 5

        expectancy_progress = _read_value(
            source, "early_termination_expectancy_progress", 0.6
        )
        if expectancy_progress is None:
            expectancy_progress = 0.6

        return cls(
            enabled=bool(_read_value(source, "enable_early_termination", False)),
            max_drawdown=_read_value(source, "early_termination_max_drawdown", None),
            min_trades=_read_value(source, "early_termination_min_trades", None),
            min_trade_check_progress=float(min_trade_check_progress),
            trade_pace_tolerance=float(trade_pace_tolerance),
            min_expectancy=_read_value(source, "early_termination_min_expectancy", None),
            expectancy_min_trades=int(expectancy_min_trades),
            expectancy_progress=float(expectancy_progress),
        )

    def to_strategy_params(self) -> dict[str, Any]:
        """strategy params 互換の legacy フィールドに変換する。"""
        return {
            "enable_early_termination": self.enabled,
            "early_termination_max_drawdown": self.max_drawdown,
            "early_termination_min_trades": self.min_trades,
            "early_termination_min_trade_check_progress": self.min_trade_check_progress,
            "early_termination_trade_pace_tolerance": self.trade_pace_tolerance,
            "early_termination_min_expectancy": self.min_expectancy,
            "early_termination_expectancy_min_trades": self.expectancy_min_trades,
            "early_termination_expectancy_progress": self.expectancy_progress,
        }


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
class MutationConfig:
    """突然変異関連設定。"""

    rate: float = 0.1
    crossover_field_selection_probability: float = 0.5
    indicator_param_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
    risk_param_range: List[float] = field(default_factory=lambda: [0.9, 1.1])
    indicator_add_delete_probability: float = 0.3
    indicator_add_vs_delete_probability: float = 0.5
    condition_change_multiplier: float = 1.0
    condition_selection_probability: float = 0.5
    condition_operator_switch_probability: float = 0.2
    tpsl_gene_creation_multiplier: float = 0.2
    position_sizing_gene_creation_multiplier: float = 0.2
    adaptive_variance_threshold: float = 0.001
    adaptive_decrease_multiplier: float = 0.8
    adaptive_increase_multiplier: float = 1.2
    valid_condition_operators: List[str] = field(
        default_factory=lambda: [
            ">",
            "<",
            ">=",
            "<=",
            "==",
            "!=",
            "CROSS_UP",
            "CROSS_DOWN",
        ]
    )

    @classmethod
    def from_dict(cls, data: dict) -> "MutationConfig":
        """
        辞書からMutationConfigインスタンスを生成

        Args:
            data: 設定値を含む辞書。未知のキーは無視されます。

        Returns:
            初期化されたMutationConfigインスタンス
        """
        filtered = _filter_known_fields(cls, data, "MutationConfig")
        return cls(**filtered)


@dataclass
class EvaluationConfig:
    """評価・検証関連設定。"""

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
    def from_dict(cls, data: dict) -> "EvaluationConfig":
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
            filtered["early_termination_settings"] = EarlyTerminationSettings.from_source(
                settings
            )
        return cls(**filtered)


@dataclass
class HybridConfig:
    """ハイブリッドGA+ML関連設定。"""

    mode: bool = False
    model_type: str = "lightgbm"
    model_types: Optional[List[str]] = None
    volatility_gate_enabled: bool = False
    volatility_model_path: Optional[str] = None
    ml_filter_enabled: bool = False
    ml_model_path: Optional[str] = None
    preprocess_features: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "HybridConfig":
        """
        辞書からHybridConfigインスタンスを生成

        Args:
            data: 設定値を含む辞書。未知のキーは無視されます。

        Returns:
            初期化されたHybridConfigインスタンス
        """
        filtered = _filter_known_fields(cls, data, "HybridConfig")
        return cls(**filtered)


@dataclass
class TuningConfig:
    """パラメータチューニング（Optuna）関連設定。"""

    enabled: bool = True
    n_trials: int = 30
    elite_count: int = 3
    use_wfa: bool = True
    include_indicators: bool = True
    include_tpsl: bool = True
    include_thresholds: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "TuningConfig":
        """
        辞書からTuningConfigインスタンスを生成

        Args:
            data: 設定値を含む辞書。未知のキーは無視されます。

        Returns:
            初期化されたTuningConfigインスタンス
        """
        filtered = _filter_known_fields(cls, data, "TuningConfig")
        return cls(**filtered)


@dataclass
class TwoStageSelectionConfig:
    """二段階選抜関連設定。"""

    enabled: bool = True
    elite_count: int = 3
    candidate_pool_size: int = 5
    min_pass_rate: float = 0.5

    @classmethod
    def from_dict(cls, data: dict) -> "TwoStageSelectionConfig":
        """
        辞書からTwoStageSelectionConfigインスタンスを生成

        Args:
            data: 設定値を含む辞書。未知のキーは無視されます。

        Returns:
            初期化されたTwoStageSelectionConfigインスタンス
        """
        filtered = _filter_known_fields(cls, data, "TwoStageSelectionConfig")
        return cls(**filtered)


@dataclass
class RobustnessConfig:
    """robustness 評価関連設定。"""

    validation_symbols: Optional[List[str]] = None
    regime_windows: List[dict] = field(default_factory=list)
    stress_slippage: List[float] = field(default_factory=list)
    stress_commission_multipliers: List[float] = field(default_factory=list)
    aggregate_method: str = "robust"

    @classmethod
    def from_dict(cls, data: dict) -> "RobustnessConfig":
        """
        辞書からRobustnessConfigインスタンスを生成

        Args:
            data: 設定値を含む辞書。未知のキーは無視されます。

        Returns:
            初期化されたRobustnessConfigインスタンス
        """
        filtered = _filter_known_fields(cls, data, "RobustnessConfig")
        return cls(**filtered)
