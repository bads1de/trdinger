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
from typing import Any, Dict, List, Optional, Set, cast

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
from .ml_filter_settings import normalize_ml_gate_fields
from .sub_configs import (
    EarlyTerminationSettings,
    EvaluationConfig,
    HybridConfig,
    MutationConfig,
    RobustnessConfig,
    TuningConfig,
    TwoStageSelectionConfig,
    resolve_early_termination_settings,
)

logger = logging.getLogger(__name__)


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
    crossover_field_selection_probability: float = 0.5
    indicator_param_mutation_range: List[float] = field(
        default_factory=lambda: [0.8, 1.2]
    )
    risk_param_mutation_range: List[float] = field(default_factory=lambda: [0.9, 1.1])
    indicator_add_delete_probability: float = 0.3
    indicator_add_vs_delete_probability: float = 0.5
    condition_change_probability_multiplier: float = 1.0
    condition_selection_probability: float = 0.5
    condition_operator_switch_probability: float = 0.2
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
    tpsl_gene_creation_probability_multiplier: float = 0.2
    position_sizing_gene_creation_probability_multiplier: float = 0.2
    adaptive_mutation_variance_threshold: float = 0.001
    adaptive_mutation_rate_decrease_multiplier: float = 0.8
    adaptive_mutation_rate_increase_multiplier: float = 1.2

    # パラメータ範囲
    parameter_ranges: Dict[str, List] = field(
        default_factory=lambda: cast(Dict[str, List], copy.deepcopy(GA_PARAMETER_RANGES))
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
    enable_fitness_sharing: bool = bool(
        GA_DEFAULT_FITNESS_SHARING["enable_fitness_sharing"]
    )
    sharing_radius: float = GA_DEFAULT_FITNESS_SHARING["sharing_radius"]
    sharing_alpha: float = GA_DEFAULT_FITNESS_SHARING["sharing_alpha"]
    sampling_threshold: int = int(GA_DEFAULT_FITNESS_SHARING["sampling_threshold"])
    sampling_ratio: float = GA_DEFAULT_FITNESS_SHARING["sampling_ratio"]

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
    early_termination_settings: Optional[EarlyTerminationSettings] = None
    enable_early_termination: bool = False
    early_termination_max_drawdown: Optional[float] = None
    early_termination_min_trades: Optional[int] = None
    early_termination_min_trade_check_progress: float = 0.5
    early_termination_trade_pace_tolerance: float = 0.5
    early_termination_min_expectancy: Optional[float] = None
    early_termination_expectancy_min_trades: int = 5
    early_termination_expectancy_progress: float = 0.6

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

    # MLフィルター設定
    volatility_gate_enabled: bool = False
    volatility_model_path: Optional[str] = None
    gate_quantile: float = 0.67
    ml_filter_enabled: bool = False
    ml_model_path: Optional[str] = None
    preprocess_features: bool = True  # 特徴量前処理を適用するかどうか

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
    enable_parameter_tuning: bool = (
        True  # エリート個体のパラメータチューニング有効化（デフォルト有効）
    )
    tuning_n_trials: int = 30  # Optunaの試行回数
    tuning_elite_count: int = 3  # チューニング対象エリート数
    tuning_use_wfa: bool = True  # WFA評価を使用（過学習防止）
    tuning_include_indicators: bool = True  # インジケーターパラメータを最適化
    tuning_include_tpsl: bool = True  # TPSLパラメータを最適化
    tuning_include_thresholds: bool = False  # 条件閾値を最適化

    # 二段階選抜設定
    enable_two_stage_selection: bool = True
    two_stage_elite_count: int = 3
    two_stage_candidate_pool_size: int = 5
    two_stage_min_pass_rate: float = 0.5

    # 二段階選抜用 robustness 設定
    robustness_validation_symbols: Optional[List[str]] = None
    robustness_regime_windows: List[Dict[str, str]] = field(default_factory=list)
    robustness_stress_slippage: List[float] = field(default_factory=list)
    robustness_stress_commission_multipliers: List[float] = field(default_factory=list)
    robustness_aggregate_method: str = "robust"

    # 階層的GA設定（サブGA）
    hierarchical_ga_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "population_size": 20,
            "generations": 10,
            "crossover_rate": 0.8,
            "mutation_rate": 0.2,
        }
    )

    # サブ設定（ネスト辞書からの復元用、None時はフラットフィールドを使用）
    mutation_config: Optional[MutationConfig] = None
    evaluation_config: Optional[EvaluationConfig] = None
    hybrid_config: Optional[HybridConfig] = None
    tuning_config: Optional[TuningConfig] = None
    two_stage_selection_config: Optional[TwoStageSelectionConfig] = None
    robustness_config: Optional[RobustnessConfig] = None

    def __post_init__(self) -> None:
        """
        インスタンス初期化後の後処理

        MLフィルター関連の設定値の整合性を確保します。
        volatility_gate_enabled と ml_filter_enabled の同期、
        およびモデルパスの相互補完を行います。
        """
        self._sync_runtime_fields()

    def _sync_runtime_fields(self) -> None:
        """互換性のあるフラットフィールドを正規化する。"""
        normalized = normalize_ml_gate_fields(self)
        self.volatility_gate_enabled = bool(normalized["volatility_gate_enabled"])
        self.ml_filter_enabled = bool(normalized["ml_filter_enabled"])
        self.volatility_model_path = cast(
            Optional[str], normalized["volatility_model_path"]
        )
        self.ml_model_path = cast(Optional[str], normalized["ml_model_path"])
        early_termination_settings = resolve_early_termination_settings(self)
        self.early_termination_settings = early_termination_settings
        early_termination_fields = early_termination_settings.to_strategy_params()
        self.enable_early_termination = bool(
            early_termination_fields["enable_early_termination"]
        )
        self.early_termination_max_drawdown = cast(
            Optional[float], early_termination_fields["early_termination_max_drawdown"]
        )
        self.early_termination_min_trades = cast(
            Optional[int], early_termination_fields["early_termination_min_trades"]
        )
        self.early_termination_min_trade_check_progress = float(
            early_termination_fields["early_termination_min_trade_check_progress"]
        )
        self.early_termination_trade_pace_tolerance = float(
            early_termination_fields["early_termination_trade_pace_tolerance"]
        )
        self.early_termination_min_expectancy = cast(
            Optional[float], early_termination_fields["early_termination_min_expectancy"]
        )
        self.early_termination_expectancy_min_trades = int(
            early_termination_fields["early_termination_expectancy_min_trades"]
        )
        self.early_termination_expectancy_progress = float(
            early_termination_fields["early_termination_expectancy_progress"]
        )
        self.indicator_universe_mode = normalize_indicator_universe_mode(
            self.indicator_universe_mode
        )

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

    @staticmethod
    def _restore_nested_configs(working: Dict[str, Any]) -> None:
        """ネストされた設定辞書をサブ設定クラスへ復元する。"""
        if "mutation_config" in working and isinstance(
            working["mutation_config"], dict
        ):
            working["mutation_config"] = MutationConfig.from_dict(
                working["mutation_config"]
            )
        if "evaluation_config" in working and isinstance(
            working["evaluation_config"], dict
        ):
            working["evaluation_config"] = EvaluationConfig.from_dict(
                working["evaluation_config"]
            )
        if "early_termination_settings" in working and isinstance(
            working["early_termination_settings"], dict
        ):
            working["early_termination_settings"] = EarlyTerminationSettings.from_dict(
                working["early_termination_settings"]
            )
        if "hybrid_config" in working and isinstance(working["hybrid_config"], dict):
            working["hybrid_config"] = HybridConfig.from_dict(working["hybrid_config"])
        if "tuning_config" in working and isinstance(working["tuning_config"], dict):
            working["tuning_config"] = TuningConfig.from_dict(working["tuning_config"])
        if "two_stage_selection_config" in working and isinstance(
            working["two_stage_selection_config"], dict
        ):
            working["two_stage_selection_config"] = TwoStageSelectionConfig.from_dict(
                working["two_stage_selection_config"]
            )
        if "robustness_config" in working and isinstance(
            working["robustness_config"], dict
        ):
            working["robustness_config"] = RobustnessConfig.from_dict(
                working["robustness_config"]
            )

    @staticmethod
    def _apply_two_stage_overrides(
        working: Dict[str, Any], provided_keys: Set[str]
    ) -> None:
        """two-stage 設定のネスト値をフラットフィールドへ反映する。"""
        two_stage_config = working.get("two_stage_selection_config")
        if not isinstance(two_stage_config, TwoStageSelectionConfig):
            return

        if "enable_two_stage_selection" not in provided_keys:
            working["enable_two_stage_selection"] = two_stage_config.enabled
        if "two_stage_elite_count" not in provided_keys:
            working["two_stage_elite_count"] = two_stage_config.elite_count
        if "two_stage_candidate_pool_size" not in provided_keys:
            working["two_stage_candidate_pool_size"] = (
                two_stage_config.candidate_pool_size
            )
        if "two_stage_min_pass_rate" not in provided_keys:
            working["two_stage_min_pass_rate"] = two_stage_config.min_pass_rate

    @staticmethod
    def _apply_robustness_overrides(
        working: Dict[str, Any], provided_keys: Set[str]
    ) -> None:
        """robustness 設定のネスト値をフラットフィールドへ反映する。"""
        robustness_config = working.get("robustness_config")
        if not isinstance(robustness_config, RobustnessConfig):
            return

        if "robustness_validation_symbols" not in provided_keys:
            working["robustness_validation_symbols"] = (
                robustness_config.validation_symbols
            )
        if "robustness_regime_windows" not in provided_keys:
            working["robustness_regime_windows"] = robustness_config.regime_windows
        if "robustness_stress_slippage" not in provided_keys:
            working["robustness_stress_slippage"] = robustness_config.stress_slippage
        if "robustness_stress_commission_multipliers" not in provided_keys:
            working["robustness_stress_commission_multipliers"] = (
                robustness_config.stress_commission_multipliers
            )
        if "robustness_aggregate_method" not in provided_keys:
            working["robustness_aggregate_method"] = robustness_config.aggregate_method

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GAConfig":
        """
        辞書形式からGAConfigインスタンスを生成

        フラット辞書とネスト辞書（mutation_config等）の両方に対応する。
        ネスト辞書が存在する場合、サブ設定からフラットフィールドへ逆展開する。

        Args:
            data: 設定値を含む辞書

        Returns:
            初期化されたGAConfigインスタンス
        """
        field_names = {field_info.name for field_info in fields(cls)}
        unknown_keys = sorted(key for key in data.keys() if key not in field_names)
        if unknown_keys:
            raise ValueError(f"未対応の設定キーがあります: {', '.join(unknown_keys)}")

        provided_keys = {key for key, value in data.items() if value is not None}
        working = copy.deepcopy(data)
        defaults = cls._from_dict_defaults()
        for key, default_value in defaults.items():
            if key not in working or working[key] is None:
                working[key] = default_value

        cls._restore_nested_configs(working)
        cls._apply_two_stage_overrides(working, provided_keys)
        cls._apply_robustness_overrides(working, provided_keys)

        # BaseConfigのfrom_dict処理を使用
        instance = cast(GAConfig, super().from_dict(working))
        instance._sync_runtime_fields()
        return instance

