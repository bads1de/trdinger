"""GAConfig ネスト設定。

GAConfig にぶら下がる設定 dataclass 群を定義する。
"""

import copy
import logging
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

from ..constants import GA_DEFAULT_CONFIG, GA_MUTATION_SETTINGS, OPERATORS

if TYPE_CHECKING:
    from .ga_config import GAConfig

logger = logging.getLogger(__name__)

DEFAULT_EARLY_TERMINATION_VALUES = {
    "enabled": True,
    "max_drawdown": 0.15,
    "min_trades": 30,
    "min_trade_check_progress": 0.33,
    "trade_pace_tolerance": 0.5,
    "min_expectancy": -0.05,
    "expectancy_min_trades": 10,
    "expectancy_progress": 0.1,
}


def _read_value(source: object, key: str, default: object = None) -> object:
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


class NestedConfigMixin:
    """ネスト設定 dataclass の共通変換処理。"""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        filtered = _filter_known_fields(cls, dict(data), cls.__name__)
        return cls(**filtered)


@dataclass(frozen=True)
class EarlyTerminationSettings(NestedConfigMixin):
    """早期終了の正規化済み設定。"""

    enabled: bool = True
    max_drawdown: Optional[float] = 0.15
    min_trades: Optional[int] = 30
    min_trade_check_progress: float = 0.33
    trade_pace_tolerance: float = 0.5
    min_expectancy: Optional[float] = -0.05
    expectancy_min_trades: int = 10
    expectancy_progress: float = 0.1

    @classmethod
    def from_source(cls, source: Mapping[str, Any]) -> "EarlyTerminationSettings":
        """dict / オブジェクトのどちらからでも設定を生成する。"""
        if isinstance(source, cls):
            return source
        if isinstance(source, Mapping):
            return cls.from_dict(source)

        filtered: Dict[str, Any] = {
            field_name: _read_value(source, field_name, default)
            for field_name, default in DEFAULT_EARLY_TERMINATION_VALUES.items()
        }
        return cls(**filtered)


@dataclass
class MutationConfig(NestedConfigMixin):
    """突然変異関連設定。"""

    rate: float = float(GA_DEFAULT_CONFIG["mutation_rate"])
    crossover_field_selection_probability: float = float(
        GA_MUTATION_SETTINGS["crossover_field_selection_probability"]
    )
    indicator_param_range: List[float] = field(
        default_factory=lambda: list(
            GA_MUTATION_SETTINGS["indicator_param_mutation_range"]
        )
    )
    risk_param_range: List[float] = field(
        default_factory=lambda: list(GA_MUTATION_SETTINGS["risk_param_mutation_range"])
    )
    indicator_add_delete_probability: float = float(
        GA_MUTATION_SETTINGS["indicator_add_delete_probability"]
    )
    indicator_add_vs_delete_probability: float = float(
        GA_MUTATION_SETTINGS["indicator_add_vs_delete_probability"]
    )
    condition_change_multiplier: float = float(
        GA_MUTATION_SETTINGS["condition_change_probability_multiplier"]
    )
    condition_selection_probability: float = float(
        GA_MUTATION_SETTINGS["condition_selection_probability"]
    )
    condition_operator_switch_probability: float = float(
        GA_MUTATION_SETTINGS["condition_operator_switch_probability"]
    )
    tpsl_gene_creation_multiplier: float = float(
        GA_MUTATION_SETTINGS["tpsl_gene_creation_probability_multiplier"]
    )
    position_sizing_gene_creation_multiplier: float = float(
        GA_MUTATION_SETTINGS["position_sizing_gene_creation_probability_multiplier"]
    )
    entry_gene_creation_multiplier: float = float(
        GA_MUTATION_SETTINGS["entry_gene_creation_probability_multiplier"]
    )
    exit_gene_creation_multiplier: float = float(
        GA_MUTATION_SETTINGS["exit_gene_creation_probability_multiplier"]
    )
    adaptive_variance_threshold: float = float(
        GA_MUTATION_SETTINGS["adaptive_mutation_variance_threshold"]
    )
    adaptive_decrease_multiplier: float = float(
        GA_MUTATION_SETTINGS["adaptive_mutation_rate_decrease_multiplier"]
    )
    adaptive_increase_multiplier: float = float(
        GA_MUTATION_SETTINGS["adaptive_mutation_rate_increase_multiplier"]
    )
    valid_condition_operators: List[str] = field(
        default_factory=lambda: OPERATORS.copy()
    )

    def bind_parent(self, parent: "GAConfig") -> None:
        """GAConfig の mutation_rate と同期するための親参照を設定する。"""
        object.__setattr__(self, "_parent_ga_config", parent)

    def __setattr__(self, name: str, value: float) -> None:
        object.__setattr__(self, name, value)
        if name == "rate":
            self._sync_parent_rate(value)

    def __deepcopy__(self, memo: dict[int, Any]) -> "MutationConfig":
        """親参照を持ち込まずに deepcopy する。"""
        copied = type(self)(
            **{
                field_info.name: copy.deepcopy(getattr(self, field_info.name), memo)
                for field_info in fields(type(self))
            }
        )
        memo[id(self)] = copied
        return copied

    def _sync_parent_rate(self, value: float) -> None:
        """親の mutation_rate を直接更新する。"""
        parent = getattr(self, "_parent_ga_config", None)
        if parent is None:
            return

        try:
            object.__setattr__(parent, "mutation_rate", value)
        except Exception as exc:
            logger.debug("mutation_rate の親同期に失敗しました: %s", exc)


@dataclass
class EvaluationConfig(NestedConfigMixin):
    """評価・検証関連設定。"""

    enable_parallel: bool = True
    max_workers: Optional[int] = None  # Noneの場合は自動で物理コア-2, 最大8に制限
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
        filtered = _filter_known_fields(cls, dict(data), "EvaluationConfig")
        settings = filtered.get("early_termination_settings")
        if settings is None:
            filtered["early_termination_settings"] = EarlyTerminationSettings()
        elif not isinstance(settings, EarlyTerminationSettings):
            filtered["early_termination_settings"] = (
                EarlyTerminationSettings.from_source(settings)
            )
        return cls(**filtered)


@dataclass
class HybridConfig(NestedConfigMixin):
    """ハイブリッドGA+ML関連設定。"""

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
    """パラメータチューニング（Optuna）関連設定。

    インジケータと TPSL は常時最適化対象として扱い、
    ここでは試行回数と閾値最適化の切り替えだけを管理する。
    """

    enabled: bool = False  # Optunaチューニングを一時的に無効化
    n_trials: int = 30
    elite_count: int = 3
    use_wfa: bool = True
    include_thresholds: bool = False


@dataclass
class TwoStageSelectionConfig(NestedConfigMixin):
    """二段階選抜関連設定。"""

    enabled: bool = True
    elite_count: int = 3
    candidate_pool_size: int = 5
    min_pass_rate: float = 0.5


@dataclass
class RobustnessConfig(NestedConfigMixin):
    """robustness 評価関連設定。"""

    validation_symbols: Optional[List[str]] = None
    regime_windows: List[dict] = field(default_factory=list)
    stress_slippage: List[float] = field(default_factory=list)
    stress_commission_multipliers: List[float] = field(default_factory=list)
    aggregate_method: str = "robust"


__all__ = [
    "EarlyTerminationSettings",
    "MutationConfig",
    "EvaluationConfig",
    "HybridConfig",
    "TuningConfig",
    "TwoStageSelectionConfig",
    "RobustnessConfig",
]
