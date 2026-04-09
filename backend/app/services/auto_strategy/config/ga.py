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
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

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
from .ga_nested_configs import (
    EARLY_TERMINATION_LEGACY_FIELD_MAP,
    EVALUATION_LEGACY_FIELD_MAP,
    HYBRID_LEGACY_FIELD_MAP,
    MUTATION_LEGACY_FIELD_MAP,
    ROBUSTNESS_LEGACY_FIELD_MAP,
    TUNING_LEGACY_FIELD_MAP,
    TWO_STAGE_SELECTION_LEGACY_FIELD_MAP,
    EvaluationConfig,
    HybridConfig,
    MutationConfig,
    RobustnessConfig,
    TuningConfig,
    TwoStageSelectionConfig,
)
from .helpers import validate_robustness_regime_window

logger = logging.getLogger(__name__)

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
    "two_stage_candidate_pool_size": (
        "two_stage_selection_config",
        "candidate_pool_size",
    ),
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
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )

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
            (
                None
                if hybrid_config.model_types is None
                else list(hybrid_config.model_types)
            ),
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
        for flat_key, nested_key in MUTATION_LEGACY_FIELD_MAP.items():
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
        for flat_key, nested_key in EVALUATION_LEGACY_FIELD_MAP.items():
            if flat_key in working:
                evaluation_payload[nested_key] = working.pop(flat_key)
        normalized["evaluation_config"] = EvaluationConfig.from_dict(evaluation_payload)

        hybrid_payload = _coerce_mapping(working.pop("hybrid_config", None))
        for flat_key, nested_key in HYBRID_LEGACY_FIELD_MAP.items():
            if flat_key in working:
                hybrid_payload[nested_key] = working.pop(flat_key)
        normalized["hybrid_config"] = HybridConfig.from_dict(hybrid_payload)

        tuning_payload = _coerce_mapping(working.pop("tuning_config", None))
        for flat_key, nested_key in TUNING_LEGACY_FIELD_MAP.items():
            if flat_key in working:
                tuning_payload[nested_key] = working.pop(flat_key)
        normalized["tuning_config"] = TuningConfig.from_dict(tuning_payload)

        two_stage_payload = _coerce_mapping(
            working.pop("two_stage_selection_config", None)
        )
        for flat_key, nested_key in TWO_STAGE_SELECTION_LEGACY_FIELD_MAP.items():
            if flat_key in working:
                two_stage_payload[nested_key] = working.pop(flat_key)
        normalized["two_stage_selection_config"] = TwoStageSelectionConfig.from_dict(
            two_stage_payload
        )

        robustness_payload = _coerce_mapping(working.pop("robustness_config", None))
        for flat_key, nested_key in ROBUSTNESS_LEGACY_FIELD_MAP.items():
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


# === GA設定プリセット ===


class GAPresets:
    """GA設定プリセットファクトリ。"""

    @staticmethod
    def quick_scan() -> GAConfig:
        """高速スキャン用（粗い探索、短時間）。

        パラメータチューニングとWFAを無効にし、小規模な集団で
        素早く有望な戦略領域を特定する用途。
        """
        return GAConfig(
            population_size=50,
            generations=20,
            max_indicators=5,
            max_conditions=2,
            tuning_config=TuningConfig(enabled=False),
            evaluation_config=EvaluationConfig(enable_walk_forward=False),
            two_stage_selection_config=TwoStageSelectionConfig(enabled=False),
            use_seed_strategies=True,
            seed_injection_rate=0.15,
            enable_parallel_evaluation=True,
        )

    @staticmethod
    def thorough_search() -> GAConfig:
        """徹底探索用（精密な探索、長時間）。

        大規模な集団と多数の世代で広範囲を探索し、
        エリート個体に対してOptunaチューニングとWFA検証を実施する。
        """
        return GAConfig(
            population_size=200,
            generations=100,
            max_indicators=10,
            max_conditions=3,
            crossover_rate=0.85,
            mutation_rate=0.15,
            elite_size=20,
            tuning_config=TuningConfig(
                enabled=True,
                n_trials=100,
                use_wfa=True,
                elite_count=5,
            ),
            evaluation_config=EvaluationConfig(
                enable_walk_forward=True,
                wfa_n_folds=5,
                wfa_train_ratio=0.7,
            ),
            use_seed_strategies=True,
            seed_injection_rate=0.1,
            fitness_sharing={"enable_fitness_sharing": True},
            enable_parallel_evaluation=True,
            two_stage_selection_config=TwoStageSelectionConfig(
                enabled=True,
                elite_count=5,
                candidate_pool_size=12,
                min_pass_rate=0.6,
            ),
            robustness_config=RobustnessConfig(
                regime_windows=[
                    {
                        "name": "early_trend",
                        "start_date": "2024-01-01 00:00:00",
                        "end_date": "2024-02-15 00:00:00",
                    },
                    {
                        "name": "late_chop",
                        "start_date": "2024-02-15 00:00:00",
                        "end_date": "2024-04-09 00:00:00",
                    },
                ],
                stress_slippage=[0.0002, 0.0005],
                stress_commission_multipliers=[1.5],
            ),
        )

    @staticmethod
    def multi_objective() -> GAConfig:
        """多目的最適化用（NSGA-II）。

        リターン、シャープレシオ、ドローダウンを同時に最適化する。
        """
        return GAConfig(
            population_size=150,
            generations=80,
            enable_multi_objective=True,
            objectives=[
                "total_return",
                "sharpe_ratio",
                "max_drawdown",
            ],
            objective_weights=[1.0, 1.0, 1.0],
            tuning_config=TuningConfig(enabled=False),
            use_seed_strategies=True,
            fitness_sharing={"enable_fitness_sharing": True},
            enable_parallel_evaluation=True,
            two_stage_selection_config=TwoStageSelectionConfig(
                enabled=True,
                elite_count=4,
                candidate_pool_size=8,
            ),
            robustness_config=RobustnessConfig(
                stress_slippage=[0.0003],
                stress_commission_multipliers=[1.5],
            ),
        )

    @staticmethod
    def short_term() -> GAConfig:
        """短期トレード用プリセット。

        高頻度取引に適した設定。トレード頻度ペナルティを緩和し、
        より.Aggressiveなパラメータ範囲を使用する。
        """
        config = GAConfig(
            population_size=100,
            generations=50,
            max_indicators=6,
            use_seed_strategies=True,
            seed_injection_rate=0.1,
            tuning_config=TuningConfig(enabled=True, n_trials=30),
            enable_parallel_evaluation=True,
        )
        # 短期向けフィットネス重み
        config.fitness_weights = {
            "total_return": 0.25,
            "sharpe_ratio": 0.30,
            "max_drawdown": 0.15,
            "win_rate": 0.10,
            "balance_score": 0.10,
            "ulcer_index_penalty": 0.05,
            "trade_frequency_penalty": 0.05,
        }
        return config

    @staticmethod
    def long_term() -> GAConfig:
        """長期トレード用プリセット。

        低頻度取引に適した設定。ドローダウンとUlcer Indexの
        ペナルティを強化し、安定性を重視する。
        """
        config = GAConfig(
            population_size=150,
            generations=80,
            max_indicators=8,
            max_conditions=3,
            use_seed_strategies=True,
            seed_injection_rate=0.1,
            tuning_config=TuningConfig(enabled=True, n_trials=50, elite_count=4),
            evaluation_config=EvaluationConfig(enable_walk_forward=True, wfa_n_folds=4),
            enable_parallel_evaluation=True,
            two_stage_selection_config=TwoStageSelectionConfig(
                enabled=True,
                elite_count=4,
                candidate_pool_size=8,
            ),
            robustness_config=RobustnessConfig(
                regime_windows=[
                    {
                        "name": "cycle_a",
                        "start_date": "2024-01-01 00:00:00",
                        "end_date": "2024-02-20 00:00:00",
                    },
                    {
                        "name": "cycle_b",
                        "start_date": "2024-02-20 00:00:00",
                        "end_date": "2024-04-09 00:00:00",
                    },
                ],
                stress_slippage=[0.0003],
                stress_commission_multipliers=[1.5],
            ),
        )
        # 長期向けフィットネス重み
        config.fitness_weights = {
            "total_return": 0.15,
            "sharpe_ratio": 0.25,
            "max_drawdown": 0.20,
            "win_rate": 0.10,
            "balance_score": 0.10,
            "ulcer_index_penalty": 0.15,
            "trade_frequency_penalty": 0.05,
        }
        return config

    @staticmethod
    def get_preset(name: str) -> Optional[GAConfig]:
        """プリセット名からGAConfigを取得する。

        Args:
            name: プリセット名（quick_scan, thorough_search, multi_objective,
                  short_term, long_term）

        Returns:
            GAConfig インスタンス、または不明な名前の場合はNone
        """
        presets = {
            "quick_scan": GAPresets.quick_scan,
            "thorough_search": GAPresets.thorough_search,
            "multi_objective": GAPresets.multi_objective,
            "short_term": GAPresets.short_term,
            "long_term": GAPresets.long_term,
        }
        factory = presets.get(name)
        return factory() if factory else None


# === 設定バリデーター ===


class ConfigValidator:
    """設定バリデーター"""

    VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    @staticmethod
    def validate(config: BaseConfig) -> Tuple[bool, List[str]]:
        """
        設定オブジェクトの妥当性を検証

        共通のバリデーション（必須項目、範囲チェック等）に加えて、
        クラス固有の検証（GAConfigの進化したロジック等）を実行します。

        Args:
            config: 検証対象の設定インスタンス

        Returns:
            (妥当であればTrue, エラーメッセージのリスト) のタプル
        """
        errors = ConfigValidator._validate_base(config)

        # クラスごとの追加検証
        if isinstance(config, GAConfig):
            errors.extend(ConfigValidator._validate_ga_config(config))

        # 将来的に GAConfig 以外の設定検証が必要になったらここに追加

        return len(errors) == 0, errors

    @staticmethod
    def _validate_base(config: BaseConfig) -> List[str]:
        """
        BaseConfigの設定に基づいた共通検証ロジック

        validation_rules に定義された必須フィールド、数値範囲、
        およびデータ型のチェックを行います。

        Args:
            config: 検証対象の設定インスタンス

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        try:
            errors.extend(ConfigValidator._validate_required_fields(config))
            errors.extend(ConfigValidator._validate_range_rules(config))
            errors.extend(ConfigValidator._validate_type_rules(config))

        except Exception as e:
            logger.error(f"基本検証中にエラーが発生: {e}", exc_info=True)
            errors.append(f"検証処理エラー: {e}")

        return errors

    @staticmethod
    def _validate_required_fields(config: BaseConfig) -> List[str]:
        """
        必須フィールドの存在確認

        Args:
            config: 検証対象の設定インスタンス

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        required_fields = config.validation_rules.get("required_fields", [])
        for field_name in required_fields:
            if not hasattr(config, field_name) or not getattr(config, field_name):
                errors.append(f"必須フィールド '{field_name}' が設定されていません")
        return errors

    @staticmethod
    def _validate_range_rules(config: BaseConfig) -> List[str]:
        """
        数値範囲の検証

        Args:
            config: 検証対象の設定インスタンス

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        range_rules = config.validation_rules.get("ranges", {})
        for field_name, (min_val, max_val) in range_rules.items():
            if hasattr(config, field_name):
                value = getattr(config, field_name)
                if isinstance(value, (int, float)) and not (
                    min_val <= value <= max_val
                ):
                    errors.append(
                        f"'{field_name}' は {min_val} から {max_val} の範囲で設定してください"
                    )
        return errors

    @staticmethod
    def _validate_type_rules(config: BaseConfig) -> List[str]:
        """
        データ型の検証

        Args:
            config: 検証対象の設定インスタンス

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        type_rules = config.validation_rules.get("types", {})
        for field_name, expected_type in type_rules.items():
            if hasattr(config, field_name):
                value = getattr(config, field_name)
                if value is not None and not isinstance(value, expected_type):
                    errors.append(
                        f"'{field_name}' は {expected_type.__name__} 型である必要があります"
                    )
        return errors

    @staticmethod
    def _validate_ga_config(config: GAConfig) -> List[str]:
        """
        GAConfig固有の検証を実行

        Args:
            config: 検証対象のGAConfigインスタンス

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        errors.extend(ConfigValidator._validate_ga_evolution_settings(config))
        errors.extend(ConfigValidator._validate_ga_oos_settings(config))
        errors.extend(ConfigValidator._validate_ga_fitness_settings(config))
        errors.extend(ConfigValidator._validate_ga_parameter_settings(config))
        errors.extend(ConfigValidator._validate_ga_execution_settings(config))
        errors.extend(ConfigValidator._validate_ga_multi_fidelity_settings(config))
        errors.extend(ConfigValidator._validate_ga_early_termination_settings(config))
        errors.extend(ConfigValidator._validate_ga_two_stage_settings(config))
        errors.extend(ConfigValidator._validate_ga_robustness_settings(config))
        return errors

    @staticmethod
    def _validate_numeric_range(
        val, min_v, max_v, name, is_int: bool = True
    ) -> List[str]:
        """
        汎用的な数値範囲検証

        Args:
            val: 検証対象の値
            min_v: 最小値
            max_v: 最大値
            name: パラメータ名（エラー表示用）
            is_int: 整数として検証するかどうか

        Returns:
            エラーメッセージのリスト
        """
        try:
            if not isinstance(val, (int, float)):
                return [f"{name}は数値である必要があります"]
            if not (min_v <= val <= max_v):
                if is_int:
                    if val > max_v:
                        return [
                            f"{name}は{max_v}以下である必要があります（パフォーマンス上の制約）"
                        ]
                    return [f"{name}は正の整数である必要があります"]
                return [f"{name}は{min_v}-{max_v}の範囲である必要があります"]
            return []
        except (TypeError, ValueError):
            return [f"{name}は数値である必要があります"]

    @staticmethod
    def _validate_ga_evolution_settings(config: GAConfig) -> List[str]:
        """
        GA進化パラメータ（個体数、世代数、交叉率、突然変異率等）の検証

        Args:
            config: 検証対象のGAConfigインスタンス

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        errors.extend(
            ConfigValidator._validate_numeric_range(
                config.population_size, 1, 1000, "個体数"
            )
        )
        errors.extend(
            ConfigValidator._validate_numeric_range(
                config.generations, 1, 500, "世代数"
            )
        )
        errors.extend(
            ConfigValidator._validate_numeric_range(
                config.crossover_rate, 0, 1, "交叉率", False
            )
        )
        errors.extend(
            ConfigValidator._validate_numeric_range(
                config.mutation_rate, 0, 1, "突然変異率", False
            )
        )

        if isinstance(config.elite_size, (int, float)) and isinstance(
            config.population_size, (int, float)
        ):
            if config.elite_size < 0 or config.elite_size >= config.population_size:
                errors.append("エリート保存数は0以上、個体数未満である必要があります")
        else:
            errors.append("elite_size と population_size は数値である必要があります")

        return errors

    @staticmethod
    def _validate_ga_oos_settings(config: GAConfig) -> List[str]:
        """
        OOS（Out-of-Sample）検証設定の検証

        Args:
            config: 検証対象のGAConfigインスタンス

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        if (
            not isinstance(config.oos_split_ratio, (int, float))
            or not 0.0 <= config.oos_split_ratio < 1.0
        ):
            errors.append("OOS分割比率は0.0以上1.0未満である必要があります")
        return errors

    @staticmethod
    def _validate_ga_fitness_settings(config: GAConfig) -> List[str]:
        """
        フィットネス計算設定（重み、メトリクス、多目的最適化等）の検証

        Args:
            config: 検証対象のGAConfigインスタンス

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        fitness_weights = getattr(config, "fitness_weights", {})
        if not isinstance(fitness_weights, dict):
            errors.append("fitness_weights は辞書である必要があります")
            return errors

        weights_are_numeric = not any(
            not isinstance(weight, (int, float)) for weight in fitness_weights.values()
        )
        if not weights_are_numeric:
            errors.append("フィットネス重みは数値である必要があります")
        else:
            if abs(sum(fitness_weights.values()) - 1.0) > 0.01:
                errors.append("フィットネス重みの合計は1.0である必要があります")

        required_metrics = {"total_return", "sharpe_ratio", "max_drawdown", "win_rate"}
        missing_metrics = required_metrics - set(fitness_weights.keys())
        if missing_metrics:
            errors.append(f"必要なメトリクスが不足しています: {missing_metrics}")

        if (
            not isinstance(config.primary_metric, str)
            or config.primary_metric not in fitness_weights
        ):
            errors.append(
                f"プライマリメトリクス '{config.primary_metric}' がフィットネス重みに含まれていません"
            )

        if "prediction_score" in fitness_weights:
            errors.append(
                "prediction_score はボラ回帰化に伴い fitness_weights ではサポートされません"
            )

        objectives = getattr(config, "objectives", None)
        if objectives is None:
            objectives = []
        if "prediction_score" in objectives:
            errors.append(
                "prediction_score はボラ回帰化に伴い objectives ではサポートされません"
            )

        return errors

    @staticmethod
    def _validate_parameter_ranges(parameter_ranges: Any) -> List[str]:
        """
        パラメータ探索範囲設定の検証

        Args:
            parameter_ranges: パラメータ名と [min, max] リストの辞書

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        if not isinstance(parameter_ranges, dict):
            errors.append("パラメータ範囲は辞書である必要があります")
            return errors

        for param, value_range in parameter_ranges.items():
            if not isinstance(value_range, list) or len(value_range) != 2:
                errors.append(
                    f"パラメータ '{param}' の範囲は [min, max] の形式である必要があります"
                )
            else:
                try:
                    if value_range[0] >= value_range[1]:
                        errors.append(
                            f"パラメータ '{param}' の最小値は最大値より小さい必要があります"
                        )
                except TypeError:
                    errors.append(
                        f"パラメータ '{param}' の最小値は最大値より小さい必要があります"
                    )
        return errors

    @staticmethod
    def _validate_ga_parameter_settings(config: GAConfig) -> List[str]:
        """
        GA戦略パラメータ（指標数、探索範囲、ログレベル等）の検証

        Args:
            config: 検証対象のGAConfigインスタンス

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        errors.extend(
            ConfigValidator._validate_numeric_range(
                config.max_indicators, 1, 10, "最大指標数"
            )
        )
        errors.extend(
            ConfigValidator._validate_parameter_ranges(config.parameter_ranges)
        )

        if (
            not isinstance(config.log_level, str)
            or config.log_level not in ConfigValidator.VALID_LOG_LEVELS
        ):
            errors.append(
                f"無効なログレベル: {config.log_level}. 有効な値: {{'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}}"
            )

        return errors

    @staticmethod
    def _validate_ga_execution_settings(config: GAConfig) -> List[str]:
        """
        GA実行環境設定（並列プロセス数等）の検証

        Args:
            config: 検証対象のGAConfigインスタンス

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        if config.parallel_processes is not None:
            if (
                not isinstance(config.parallel_processes, (int, float))
                or config.parallel_processes <= 0
            ):
                errors.append("並列プロセス数は正の整数である必要があります")
            elif config.parallel_processes > 32:
                errors.append("並列プロセス数は32以下である必要があります")
        return errors

    @staticmethod
    def _validate_ga_multi_fidelity_settings(config: GAConfig) -> List[str]:
        """multi-fidelity 評価設定の検証"""
        errors = []
        if not getattr(config, "enable_multi_fidelity_evaluation", False):
            return errors

        if (
            not isinstance(config.multi_fidelity_window_ratio, (int, float))
            or not 0.0 < float(config.multi_fidelity_window_ratio) <= 1.0
        ):
            errors.append(
                "multi_fidelity_window_ratio は0より大きく1.0以下である必要があります"
            )

        if (
            not isinstance(config.multi_fidelity_oos_ratio, (int, float))
            or not 0.0 < float(config.multi_fidelity_oos_ratio) < 1.0
        ):
            errors.append(
                "multi_fidelity_oos_ratio は0より大きく1.0未満である必要があります"
            )

        if (
            not isinstance(config.multi_fidelity_candidate_ratio, (int, float))
            or not 0.0 < float(config.multi_fidelity_candidate_ratio) <= 1.0
        ):
            errors.append(
                "multi_fidelity_candidate_ratio は0より大きく1.0以下である必要があります"
            )

        if (
            not isinstance(config.multi_fidelity_min_candidates, (int, float))
            or int(config.multi_fidelity_min_candidates) <= 0
        ):
            errors.append(
                "multi_fidelity_min_candidates は正の整数である必要があります"
            )

        return errors

    @staticmethod
    def _validate_ga_early_termination_settings(config: GAConfig) -> List[str]:
        """早期打ち切り設定の検証"""
        errors = []
        settings = config.evaluation_config.early_termination_settings
        if not settings.enabled:
            return errors

        max_drawdown = settings.max_drawdown
        if max_drawdown is not None and (
            not isinstance(max_drawdown, (int, float))
            or not 0.0 < float(max_drawdown) <= 1.0
        ):
            errors.append(
                "early_termination_max_drawdown は0より大きく1.0以下である必要があります"
            )

        min_trades = settings.min_trades
        if min_trades is not None and (
            not isinstance(min_trades, (int, float)) or int(min_trades) <= 0
        ):
            errors.append("early_termination_min_trades は正の整数である必要があります")

        expectancy = settings.min_expectancy
        if expectancy is not None and not isinstance(expectancy, (int, float)):
            errors.append("early_termination_min_expectancy は数値である必要があります")

        expectancy_min_trades = settings.expectancy_min_trades
        if (
            not isinstance(expectancy_min_trades, (int, float))
            or int(expectancy_min_trades) <= 0
        ):
            errors.append(
                "early_termination_expectancy_min_trades は正の整数である必要があります"
            )

        for field_name in (
            "min_trade_check_progress",
            "trade_pace_tolerance",
            "expectancy_progress",
        ):
            value = getattr(settings, field_name, None)
            if not isinstance(value, (int, float)) or not 0.0 < float(value) <= 1.0:
                errors.append(
                    f"early_termination_{field_name} は0より大きく1.0以下である必要があります"
                )

        return errors

    @staticmethod
    def _validate_ga_two_stage_settings(config: GAConfig) -> List[str]:
        """
        二段階選抜設定の検証

        Args:
            config: 検証対象のGAConfigインスタンス

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        two_stage_config = config.two_stage_selection_config
        if not two_stage_config.enabled:
            return errors

        elite_count = two_stage_config.elite_count
        population_size = config.population_size
        candidate_pool_size = two_stage_config.candidate_pool_size

        if not isinstance(elite_count, (int, float)) or int(elite_count) <= 0:
            errors.append("二段階選抜エリート数は正の整数である必要があります")
        elif isinstance(population_size, (int, float)) and int(elite_count) >= int(
            population_size
        ):
            errors.append("二段階選抜エリート数は個体数未満である必要があります")

        if (
            not isinstance(candidate_pool_size, (int, float))
            or int(candidate_pool_size) <= 0
        ):
            errors.append("二段階選抜候補数は正の整数である必要があります")
        elif isinstance(elite_count, (int, float)) and int(candidate_pool_size) < int(
            elite_count
        ):
            errors.append(
                "二段階選抜候補数は二段階選抜エリート数以上である必要があります"
            )

        if (
            not isinstance(two_stage_config.min_pass_rate, (int, float))
            or not 0.0 <= float(two_stage_config.min_pass_rate) <= 1.0
        ):
            errors.append("二段階選抜 pass rate は0.0-1.0の範囲である必要があります")

        return errors

    @staticmethod
    def _validate_robustness_validation_symbols(config: GAConfig) -> List[str]:
        """
        robustness 検証用通貨ペア設定の検証

        Args:
            config: 検証対象のGAConfigインスタンス

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        validation_symbols = config.robustness_config.validation_symbols
        if validation_symbols is not None and not isinstance(validation_symbols, list):
            errors.append("robustness_validation_symbols はリストである必要があります")
        return errors

    @staticmethod
    def _validate_robustness_window(window: Any) -> List[str]:
        """
        robustness 検証用期間設定の検証

        Args:
            window: 検証期間設定を含む辞書

        Returns:
            エラーメッセージのリスト
        """
        return validate_robustness_regime_window(window)

    @staticmethod
    def _validate_robustness_regime_windows(config: GAConfig) -> List[str]:
        """
        robustness 検証用全期間リストの検証

        Args:
            config: 検証対象のGAConfigインスタンス

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        regime_windows = config.robustness_config.regime_windows
        if not isinstance(regime_windows, list):
            errors.append("robustness の regime windows はリストである必要があります")
            return errors

        for window in regime_windows:
            window_errors = ConfigValidator._validate_robustness_window(window)
            if window_errors:
                errors.extend(window_errors)
                break

        return errors

    @staticmethod
    def _validate_non_negative_numeric_list(values: Any, label: str) -> List[str]:
        """
        非負数値リストの検証

        Args:
            values: 検証対象のリスト
            label: エラー表示用ラベル

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        if not isinstance(values, list):
            errors.append(f"{label} はリストである必要があります")
            return errors

        for value in values:
            if not isinstance(value, (int, float)) or float(value) < 0.0:
                errors.append(f"{label} は0以上の数値である必要があります")
                break

        return errors

    @staticmethod
    def _validate_positive_numeric_list(values: Any, label: str) -> List[str]:
        """
        正の数値リストの検証

        Args:
            values: 検証対象のリスト
            label: エラー表示用ラベル

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        if not isinstance(values, list):
            errors.append(f"{label} はリストである必要があります")
            return errors

        for value in values:
            if not isinstance(value, (int, float)) or float(value) <= 0.0:
                errors.append(f"{label} は正の数値である必要があります")
                break

        return errors

    @staticmethod
    def _validate_aggregate_method(config: GAConfig) -> List[str]:
        """
        robustness 評価集計方法の検証

        Args:
            config: 検証対象のGAConfigインスタンス

        Returns:
            エラーメッセージのリスト
        """
        aggregate_method = config.robustness_config.aggregate_method
        if not isinstance(aggregate_method, str) or aggregate_method not in {
            "robust",
            "mean",
        }:
            return [
                "robustness_aggregate_method は {'robust', 'mean'} のいずれかである必要があります"
            ]
        return []

    @staticmethod
    def _validate_ga_robustness_settings(config: GAConfig) -> List[str]:
        """
        robustness 検証全設定の検証

        Args:
            config: 検証対象のGAConfigインスタンス

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        errors.extend(ConfigValidator._validate_robustness_validation_symbols(config))
        errors.extend(ConfigValidator._validate_robustness_regime_windows(config))
        slippage = config.robustness_config.stress_slippage
        errors.extend(
            ConfigValidator._validate_non_negative_numeric_list(
                slippage,
                "robustness の slippage",
            )
        )
        commission_multipliers = config.robustness_config.stress_commission_multipliers
        errors.extend(
            ConfigValidator._validate_positive_numeric_list(
                commission_multipliers,
                "robustness の commission multiplier",
            )
        )
        errors.extend(ConfigValidator._validate_aggregate_method(config))
        return errors
