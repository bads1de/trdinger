"""
GA実行時設定クラス

GAConfig クラスを提供します。
GAConfig は GA エンジンのランタイム設定用 dataclass です。
環境変数ベースの設定が必要な場合は auto_strategy_settings.AutoStrategyConfig を使用してください。
両者の基本パラメータのデフォルト値は ga_constants.GA_DEFAULT_CONFIG を共有しています。
"""

import copy
import logging
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Mapping, Optional, cast

from app.utils.serialization import dataclass_to_dict

from ..indicator_universe import normalize_indicator_universe_mode
from .base import BaseConfig
from .constants import (
    DEFAULT_FITNESS_CONSTRAINTS,
    DEFAULT_FITNESS_WEIGHTS,
    DEFAULT_GA_OBJECTIVE_WEIGHTS,
    DEFAULT_GA_OBJECTIVES,
    GA_DEFAULT_CONFIG,
    GA_DEFAULT_FITNESS_SHARING,
    GA_PARAMETER_RANGES,
    GA_THRESHOLD_RANGES,
)
from .sub_configs import (
    EARLY_TERMINATION_LEGACY_FIELD_MAP,
    EvaluationConfig,
    HybridConfig,
    MutationConfig,
    RobustnessConfig,
    TuningConfig,
    TwoStageSelectionConfig,
)

logger = logging.getLogger(__name__)


_MUTATION_LEGACY_FIELD_MAP = {
    "crossover_field_selection_probability": "crossover_field_selection_probability",
    "indicator_param_mutation_range": "indicator_param_range",
    "risk_param_mutation_range": "risk_param_range",
    "indicator_add_delete_probability": "indicator_add_delete_probability",
    "indicator_add_vs_delete_probability": "indicator_add_vs_delete_probability",
    "condition_change_probability_multiplier": "condition_change_multiplier",
    "condition_selection_probability": "condition_selection_probability",
    "condition_operator_switch_probability": "condition_operator_switch_probability",
    "tpsl_gene_creation_probability_multiplier": "tpsl_gene_creation_multiplier",
    "position_sizing_gene_creation_probability_multiplier": "position_sizing_gene_creation_multiplier",
    "adaptive_mutation_variance_threshold": "adaptive_variance_threshold",
    "adaptive_mutation_rate_decrease_multiplier": "adaptive_decrease_multiplier",
    "adaptive_mutation_rate_increase_multiplier": "adaptive_increase_multiplier",
    "valid_condition_operators": "valid_condition_operators",
}

_EVALUATION_LEGACY_FIELD_MAP = {
    "enable_parallel_evaluation": "enable_parallel",
    "max_evaluation_workers": "max_workers",
    "evaluation_timeout": "timeout",
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

_HYBRID_LEGACY_FIELD_MAP = {
    "hybrid_mode": "mode",
    "hybrid_model_type": "model_type",
    "hybrid_model_types": "model_types",
    "volatility_gate_enabled": "volatility_gate_enabled",
    "volatility_model_path": "volatility_model_path",
    "ml_filter_enabled": "ml_filter_enabled",
    "ml_model_path": "ml_model_path",
    "preprocess_features": "preprocess_features",
}

_TUNING_LEGACY_FIELD_MAP = {
    "enable_parameter_tuning": "enabled",
    "tuning_n_trials": "n_trials",
    "tuning_elite_count": "elite_count",
    "tuning_use_wfa": "use_wfa",
    "tuning_include_indicators": "include_indicators",
    "tuning_include_tpsl": "include_tpsl",
    "tuning_include_thresholds": "include_thresholds",
}

_TWO_STAGE_LEGACY_FIELD_MAP = {
    "enable_two_stage_selection": "enabled",
    "two_stage_elite_count": "elite_count",
    "two_stage_candidate_pool_size": "candidate_pool_size",
    "two_stage_min_pass_rate": "min_pass_rate",
}

_ROBUSTNESS_LEGACY_FIELD_MAP = {
    "robustness_validation_symbols": "validation_symbols",
    "robustness_regime_windows": "regime_windows",
    "robustness_stress_slippage": "stress_slippage",
    "robustness_stress_commission_multipliers": "stress_commission_multipliers",
    "robustness_aggregate_method": "aggregate_method",
}

_FITNESS_SHARING_LEGACY_KEYS = (
    "enable_fitness_sharing",
    "sharing_radius",
    "sharing_alpha",
    "sampling_threshold",
    "sampling_ratio",
)

_LEGACY_ATTRIBUTE_PATHS = {
    "enable_parameter_tuning": ("tuning_config", "enabled"),
    "tuning_n_trials": ("tuning_config", "n_trials"),
    "tuning_elite_count": ("tuning_config", "elite_count"),
    "tuning_use_wfa": ("tuning_config", "use_wfa"),
    "tuning_include_indicators": ("tuning_config", "include_indicators"),
    "tuning_include_tpsl": ("tuning_config", "include_tpsl"),
    "tuning_include_thresholds": ("tuning_config", "include_thresholds"),
    "enable_two_stage_selection": ("two_stage_selection_config", "enabled"),
    "two_stage_elite_count": ("two_stage_selection_config", "elite_count"),
    "two_stage_candidate_pool_size": ("two_stage_selection_config", "candidate_pool_size"),
    "two_stage_min_pass_rate": ("two_stage_selection_config", "min_pass_rate"),
    "robustness_validation_symbols": ("robustness_config", "validation_symbols"),
    "robustness_regime_windows": ("robustness_config", "regime_windows"),
    "robustness_stress_slippage": ("robustness_config", "stress_slippage"),
    "robustness_stress_commission_multipliers": (
        "robustness_config",
        "stress_commission_multipliers",
    ),
    "robustness_aggregate_method": ("robustness_config", "aggregate_method"),
    "early_termination_settings": (
        "evaluation_config",
        "early_termination_settings",
    ),
    "enable_early_termination": (
        "evaluation_config",
        "early_termination_settings",
        "enabled",
    ),
    "early_termination_max_drawdown": (
        "evaluation_config",
        "early_termination_settings",
        "max_drawdown",
    ),
    "early_termination_min_trades": (
        "evaluation_config",
        "early_termination_settings",
        "min_trades",
    ),
    "early_termination_min_trade_check_progress": (
        "evaluation_config",
        "early_termination_settings",
        "min_trade_check_progress",
    ),
    "early_termination_trade_pace_tolerance": (
        "evaluation_config",
        "early_termination_settings",
        "trade_pace_tolerance",
    ),
    "early_termination_min_expectancy": (
        "evaluation_config",
        "early_termination_settings",
        "min_expectancy",
    ),
    "early_termination_expectancy_min_trades": (
        "evaluation_config",
        "early_termination_settings",
        "expectancy_min_trades",
    ),
    "early_termination_expectancy_progress": (
        "evaluation_config",
        "early_termination_settings",
        "expectancy_progress",
    ),
    "enable_fitness_sharing": ("fitness_sharing", "enable_fitness_sharing"),
    "sharing_radius": ("fitness_sharing", "sharing_radius"),
    "sharing_alpha": ("fitness_sharing", "sharing_alpha"),
    "sampling_threshold": ("fitness_sharing", "sampling_threshold"),
    "sampling_ratio": ("fitness_sharing", "sampling_ratio"),
}


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    """dict / オブジェクトのどちらからでも浅い辞書を作る。"""
    if value is None:
        return {}

    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))

    try:
        return copy.deepcopy(dict(vars(value)))
    except TypeError:
        return {}


@dataclass
class GAConfig(BaseConfig):
    """
    実行時GA設定クラス

    GA実行時のフラット設定を管理する。
    """

    # 基本GA設定
    population_size: int = int(GA_DEFAULT_CONFIG["population_size"])
    generations: int = int(GA_DEFAULT_CONFIG["generations"])
    crossover_rate: float = GA_DEFAULT_CONFIG["crossover_rate"]
    mutation_rate: float = GA_DEFAULT_CONFIG["mutation_rate"]
    elite_size: int = int(GA_DEFAULT_CONFIG["elite_size"])
    max_indicators: int = int(GA_DEFAULT_CONFIG["max_indicators"])

    # 戦略生成制約
    min_indicators: int = 1
    min_conditions: int = 1
    max_conditions: int = 3

    # ペナルティ設定
    zero_trades_penalty: float = GA_DEFAULT_CONFIG["zero_trades_penalty"]
    constraint_violation_penalty: float = GA_DEFAULT_CONFIG[
        "constraint_violation_penalty"
    ]

    # 交叉・突然変異拡張設定
    # サブ設定: mutation_config

    # パラメータ範囲
    parameter_ranges: Dict[str, List] = field(
        default_factory=lambda: cast(
            Dict[str, List], copy.deepcopy(GA_PARAMETER_RANGES)
        )
    )
    threshold_ranges: Dict[str, List[float]] = field(
        default_factory=lambda: cast(
            Dict[str, List[float]], copy.deepcopy(GA_THRESHOLD_RANGES)
        )
    )

    # フィットネス設定
    fitness_weights: Dict[str, float] = field(
        default_factory=lambda: DEFAULT_FITNESS_WEIGHTS.copy()
    )
    fitness_constraints: Dict[str, Any] = field(
        default_factory=lambda: DEFAULT_FITNESS_CONSTRAINTS.copy()
    )

    # フィットネス共有設定
    fitness_sharing: Dict[str, Any] = field(
        default_factory=lambda: GA_DEFAULT_FITNESS_SHARING.copy()
    )

    # 多目的最適化設定
    enable_multi_objective: bool = False
    objectives: List[str] = field(default_factory=lambda: DEFAULT_GA_OBJECTIVES.copy())
    objective_weights: List[float] = field(
        default_factory=lambda: DEFAULT_GA_OBJECTIVE_WEIGHTS.copy()
    )
    dynamic_objective_reweighting: bool = True
    objective_dynamic_scalars: Dict[str, float] = field(default_factory=dict)

    # 評価設定拡張（単一目的最適化用）
    primary_metric: str = "sharpe_ratio"

    # 実行時設定
    parallel_processes: Optional[int] = None
    random_state: Optional[int] = None

    # 並列評価設定
    enable_parallel_evaluation: bool = True
    max_evaluation_workers: Optional[int] = None  # Noneの場合はCPUコア数×2
    evaluation_timeout: float = 300.0  # 個体あたりのタイムアウト秒数
    enable_multi_fidelity_evaluation: bool = False
    multi_fidelity_window_ratio: float = 0.3
    multi_fidelity_oos_ratio: float = 0.2
    multi_fidelity_candidate_ratio: float = 0.25
    multi_fidelity_min_candidates: int = 3

    # ハイブリッドGA+ML設定
    hybrid_mode: bool = False
    hybrid_model_type: str = "lightgbm"  # lightgbm, xgboost, randomforest
    hybrid_model_types: Optional[List[str]] = None  # 複数モデル平均の場合
    log_level: str = "ERROR"
    save_intermediate_results: bool = True

    # フォールバック設定
    fallback_start_date: str = "2024-01-01"
    fallback_end_date: str = "2024-04-09"

    # PurgedKFold設定（過学習対策）
    enable_purged_kfold: bool = False  # PurgedKFoldを有効にするフラグ
    purged_kfold_splits: int = 5  # 分割数
    purged_kfold_embargo: float = 0.01  # エンバーゴ率

    # 遺伝子生成設定拡張
    price_data_weight: int = 3
    volume_data_weight: int = 1
    oi_fr_data_weight: int = 1
    numeric_threshold_probability: float = 0.8
    min_compatibility_score: float = 0.8
    strict_compatibility_score: float = 0.9
    indicator_universe_mode: str = "curated"

    # TPSL関連設定拡張
    tpsl_method_constraints: Optional[List[str]] = None
    tpsl_sl_range: Optional[List[float]] = None
    tpsl_tp_range: Optional[List[float]] = None
    tpsl_rr_range: Optional[List[float]] = None
    tpsl_atr_multiplier_range: Optional[List[float]] = None

    # OOS検証設定
    oos_split_ratio: float = 0.0
    oos_fitness_weight: float = 0.5

    # Walk-Forward Analysis (WFA) 設定
    enable_walk_forward: bool = False  # WFA を有効にするか
    wfa_n_folds: int = 5  # フォールド数（ローリングウィンドウの回数）
    wfa_train_ratio: float = 0.7  # 各フォールド内での学習期間の比率（0.0-1.0）
    wfa_anchored: bool = False  # True: 学習開始点を固定（拡張WFA）、False: ローリング

    # マルチタイムフレーム（MTF）設定
    enable_multi_timeframe: bool = False
    available_timeframes: Optional[List[str]] = (
        None  # 利用可能なタイムフレーム（Noneの場合はデフォルト）
    )
    mtf_indicator_probability: float = 0.3  # MTF指標が生成される確率（0.0-1.0）

    # パラメータ範囲プリセット設定
    # 探索範囲のプリセット名（例: "short_term", "mid_term", "long_term"）
    # None の場合はデフォルト範囲を使用
    parameter_range_preset: Optional[str] = None

    # シード戦略設定（ハイブリッド初期化）
    # 実戦的な戦略テンプレートを初期集団に注入し、探索効率を向上させる
    use_seed_strategies: bool = True  # シード戦略を使用するか
    seed_injection_rate: float = 0.1  # 初期集団のうちシードで置き換える割合（0.0-1.0）

    # パラメータチューニング設定（GA×Optunaハイブリッド）
    # サブ設定: tuning_config

    # 二段階選抜設定
    # サブ設定: two_stage_selection_config

    # 二段階選抜用 robustness 設定
    # サブ設定: robustness_config

    # 階層的GA設定（サブGA）
    hierarchical_ga_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "population_size": 20,
            "generations": 10,
            "crossover_rate": 0.8,
            "mutation_rate": 0.2,
        }
    )

    # サブ設定（ネスト辞書からの復元用）
    mutation_config: MutationConfig = field(default_factory=MutationConfig)
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)
    hybrid_config: HybridConfig = field(default_factory=HybridConfig)
    tuning_config: TuningConfig = field(default_factory=TuningConfig)
    two_stage_selection_config: TwoStageSelectionConfig = field(
        default_factory=TwoStageSelectionConfig
    )
    robustness_config: RobustnessConfig = field(default_factory=RobustnessConfig)

    def __init__(self, **data: Any) -> None:
        """legacy flat keys も受け付ける手動初期化器。"""
        normalized = self._normalize_input_data(data)
        defaults = self._from_dict_defaults()
        defaults.update(normalized)

        for key, value in defaults.items():
            object.__setattr__(self, key, value)

        self.__post_init__()

    def __post_init__(self) -> None:
        """初期化後に呼ばれる整合性同期フック。"""
        self._sync_runtime_fields()

    def __getattr__(self, name: str) -> Any:
        """legacy 属性名を新しいネスト設定へフォールバックする。"""
        path = _LEGACY_ATTRIBUTE_PATHS.get(name)
        if path is None:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

        value: Any = self
        for segment in path:
            if isinstance(value, dict):
                value = value.get(segment)
            else:
                value = getattr(value, segment)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        """legacy 属性名への代入をネスト設定へ反映する。"""
        path = _LEGACY_ATTRIBUTE_PATHS.get(name)
        if path is None:
            object.__setattr__(self, name, value)
            return

        try:
            target: Any = object.__getattribute__(self, path[0])
            for segment in path[1:-1]:
                if isinstance(target, dict):
                    target = target.get(segment)
                else:
                    target = getattr(target, segment)

            last_segment = path[-1]
            if isinstance(target, dict):
                target[last_segment] = value
            else:
                object.__setattr__(target, last_segment, value)
        except Exception:
            object.__setattr__(self, name, value)

    def _sync_runtime_fields(self) -> None:
        """互換性のあるフィールドを正規化する。"""
        object.__setattr__(
            self,
            "indicator_universe_mode",
            normalize_indicator_universe_mode(self.indicator_universe_mode),
        )

        if not isinstance(self.fitness_sharing, dict):
            object.__setattr__(
                self,
                "fitness_sharing",
                copy.deepcopy(GA_DEFAULT_FITNESS_SHARING),
            )
        else:
            merged_fitness_sharing = copy.deepcopy(GA_DEFAULT_FITNESS_SHARING)
            merged_fitness_sharing.update(self.fitness_sharing)
            object.__setattr__(self, "fitness_sharing", merged_fitness_sharing)

        if isinstance(self.mutation_config, dict):
            object.__setattr__(
                self,
                "mutation_config",
                MutationConfig.from_dict(self.mutation_config),
            )
        if isinstance(self.evaluation_config, dict):
            object.__setattr__(
                self,
                "evaluation_config",
                EvaluationConfig.from_dict(self.evaluation_config),
            )
        if isinstance(self.hybrid_config, dict):
            object.__setattr__(
                self, "hybrid_config", HybridConfig.from_dict(self.hybrid_config)
            )
        if isinstance(self.tuning_config, dict):
            object.__setattr__(
                self, "tuning_config", TuningConfig.from_dict(self.tuning_config)
            )
        if isinstance(self.two_stage_selection_config, dict):
            object.__setattr__(
                self,
                "two_stage_selection_config",
                TwoStageSelectionConfig.from_dict(self.two_stage_selection_config),
            )
        if isinstance(self.robustness_config, dict):
            object.__setattr__(
                self,
                "robustness_config",
                RobustnessConfig.from_dict(self.robustness_config),
            )

        evaluation_config = self.evaluation_config
        object.__setattr__(
            self,
            "enable_parallel_evaluation",
            bool(evaluation_config.enable_parallel),
        )
        object.__setattr__(
            self,
            "max_evaluation_workers",
            evaluation_config.max_workers,
        )
        object.__setattr__(self, "evaluation_timeout", evaluation_config.timeout)
        object.__setattr__(
            self,
            "enable_multi_fidelity_evaluation",
            bool(evaluation_config.enable_multi_fidelity_evaluation),
        )
        object.__setattr__(
            self,
            "multi_fidelity_window_ratio",
            evaluation_config.multi_fidelity_window_ratio,
        )
        object.__setattr__(
            self,
            "multi_fidelity_oos_ratio",
            evaluation_config.multi_fidelity_oos_ratio,
        )
        object.__setattr__(
            self,
            "multi_fidelity_candidate_ratio",
            evaluation_config.multi_fidelity_candidate_ratio,
        )
        object.__setattr__(
            self,
            "multi_fidelity_min_candidates",
            evaluation_config.multi_fidelity_min_candidates,
        )
        object.__setattr__(self, "oos_split_ratio", evaluation_config.oos_split_ratio)
        object.__setattr__(
            self, "oos_fitness_weight", evaluation_config.oos_fitness_weight
        )
        object.__setattr__(
            self,
            "enable_walk_forward",
            bool(evaluation_config.enable_walk_forward),
        )
        object.__setattr__(self, "wfa_n_folds", evaluation_config.wfa_n_folds)
        object.__setattr__(self, "wfa_train_ratio", evaluation_config.wfa_train_ratio)
        object.__setattr__(self, "wfa_anchored", evaluation_config.wfa_anchored)

        hybrid_config = self.hybrid_config
        object.__setattr__(self, "hybrid_mode", bool(hybrid_config.mode))
        object.__setattr__(self, "hybrid_model_type", hybrid_config.model_type)
        object.__setattr__(
            self,
            "hybrid_model_types",
            None if hybrid_config.model_types is None else list(hybrid_config.model_types),
        )
        object.__setattr__(
            self,
            "volatility_gate_enabled",
            bool(hybrid_config.volatility_gate_enabled),
        )
        object.__setattr__(
            self,
            "volatility_model_path",
            hybrid_config.volatility_model_path,
        )
        object.__setattr__(
            self,
            "ml_filter_enabled",
            bool(hybrid_config.ml_filter_enabled),
        )
        object.__setattr__(self, "ml_model_path", hybrid_config.ml_model_path)
        object.__setattr__(
            self,
            "preprocess_features",
            bool(hybrid_config.preprocess_features),
        )

        mutation_config = self.mutation_config
        object.__setattr__(self, "mutation_rate", mutation_config.rate)

    def to_dict(self) -> Dict[str, Any]:
        """
        設定オブジェクトを辞書形式に変換

        DBへの保存やJSONシリアライズのために使用されます。

        Returns:
            設定値を含む辞書
        """
        return cast(Dict[str, Any], dataclass_to_dict(self))

    @classmethod
    def _from_dict_defaults(cls) -> Dict[str, Any]:
        """from_dict 用のデフォルト値を生成する。"""
        return cast(Dict[str, Any], copy.deepcopy(cls.get_default_values_from_fields()))

    @classmethod
    def _normalize_input_data(cls, data: Mapping[str, Any]) -> Dict[str, Any]:
        """legacy flat keys を含む入力を正規化する。"""
        working = copy.deepcopy(dict(data))
        normalized = cls._from_dict_defaults()

        mutation_payload = _coerce_mapping(working.pop("mutation_config", None))
        if "mutation_rate" in working:
            mutation_payload["rate"] = working.pop("mutation_rate")
        for flat_key, nested_key in _MUTATION_LEGACY_FIELD_MAP.items():
            if flat_key in working:
                mutation_payload[nested_key] = working.pop(flat_key)
        normalized["mutation_config"] = MutationConfig.from_dict(mutation_payload)

        evaluation_payload = _coerce_mapping(working.pop("evaluation_config", None))
        early_termination_payload = _coerce_mapping(
            evaluation_payload.pop("early_termination_settings", None)
        )
        if "early_termination_settings" in working:
            early_termination_payload.update(
                _coerce_mapping(working.pop("early_termination_settings", None))
            )
        for flat_key, nested_key in EARLY_TERMINATION_LEGACY_FIELD_MAP.items():
            if flat_key in working:
                early_termination_payload[nested_key] = working.pop(flat_key)
        if early_termination_payload:
            evaluation_payload["early_termination_settings"] = early_termination_payload
        for flat_key, nested_key in _EVALUATION_LEGACY_FIELD_MAP.items():
            if flat_key in working:
                evaluation_payload[nested_key] = working.pop(flat_key)
        normalized["evaluation_config"] = EvaluationConfig.from_dict(evaluation_payload)

        hybrid_payload = _coerce_mapping(working.pop("hybrid_config", None))
        for flat_key, nested_key in _HYBRID_LEGACY_FIELD_MAP.items():
            if flat_key in working:
                hybrid_payload[nested_key] = working.pop(flat_key)
        normalized["hybrid_config"] = HybridConfig.from_dict(hybrid_payload)

        tuning_payload = _coerce_mapping(working.pop("tuning_config", None))
        for flat_key, nested_key in _TUNING_LEGACY_FIELD_MAP.items():
            if flat_key in working:
                tuning_payload[nested_key] = working.pop(flat_key)
        normalized["tuning_config"] = TuningConfig.from_dict(tuning_payload)

        two_stage_payload = _coerce_mapping(working.pop("two_stage_selection_config", None))
        for flat_key, nested_key in _TWO_STAGE_LEGACY_FIELD_MAP.items():
            if flat_key in working:
                two_stage_payload[nested_key] = working.pop(flat_key)
        normalized["two_stage_selection_config"] = TwoStageSelectionConfig.from_dict(
            two_stage_payload
        )

        robustness_payload = _coerce_mapping(working.pop("robustness_config", None))
        for flat_key, nested_key in _ROBUSTNESS_LEGACY_FIELD_MAP.items():
            if flat_key in working:
                robustness_payload[nested_key] = working.pop(flat_key)
        normalized["robustness_config"] = RobustnessConfig.from_dict(robustness_payload)

        fitness_sharing_payload = copy.deepcopy(normalized["fitness_sharing"])
        if "fitness_sharing" in working:
            fitness_sharing_payload.update(
                _coerce_mapping(working.pop("fitness_sharing", None))
            )
        for legacy_key in _FITNESS_SHARING_LEGACY_KEYS:
            if legacy_key in working:
                fitness_sharing_payload[legacy_key] = working.pop(legacy_key)
        normalized["fitness_sharing"] = fitness_sharing_payload

        field_names = {field_info.name for field_info in fields(cls)}
        for key in list(working.keys()):
            if key in field_names:
                normalized[key] = working.pop(key)

        if working:
            unknown_keys = sorted(working.keys())
            raise ValueError(f"未対応の設定キーがあります: {', '.join(unknown_keys)}")

        return normalized

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GAConfig":
        """
        辞書形式からGAConfigインスタンスを生成

        legacy のフラット設定とネスト辞書の両方に対応する。
        """
        return cls(**copy.deepcopy(data))
