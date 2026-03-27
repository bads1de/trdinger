"""
GA実行時設定クラス

GAConfig クラスを提供します。
GAConfig は GA エンジンのランタイム設定用 dataclass です。
環境変数ベースの設定が必要な場合は auto_strategy_settings.AutoStrategyConfig を使用してください。
両者の基本パラメータのデフォルト値は ga_constants.GA_DEFAULT_CONFIG を共有しています。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, cast

from app.utils.serialization import dataclass_to_dict

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
    EvaluationConfig,
    HybridConfig,
    MutationConfig,
    TuningConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class GAConfig(BaseConfig):
    """
    実行時GA設定クラス

    GA実行時のフラット設定を管理する。
    """

    # 基本GA設定
    population_size: int = GA_DEFAULT_CONFIG["population_size"]
    generations: int = GA_DEFAULT_CONFIG["generations"]
    crossover_rate: float = GA_DEFAULT_CONFIG["crossover_rate"]
    mutation_rate: float = GA_DEFAULT_CONFIG["mutation_rate"]
    elite_size: int = GA_DEFAULT_CONFIG["elite_size"]
    max_indicators: int = GA_DEFAULT_CONFIG["max_indicators"]

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
        default_factory=lambda: GA_PARAMETER_RANGES.copy()
    )
    threshold_ranges: Dict[str, List[float]] = field(
        default_factory=lambda: GA_THRESHOLD_RANGES.copy()
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
    enable_fitness_sharing: bool = GA_DEFAULT_FITNESS_SHARING["enable_fitness_sharing"]
    sharing_radius: float = GA_DEFAULT_FITNESS_SHARING["sharing_radius"]
    sharing_alpha: float = GA_DEFAULT_FITNESS_SHARING["sharing_alpha"]
    sampling_threshold: int = GA_DEFAULT_FITNESS_SHARING["sampling_threshold"]
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

    # ハイブリッドGA+ML設定
    hybrid_mode: bool = False
    hybrid_model_type: str = "lightgbm"  # lightgbm, xgboost, randomforest
    hybrid_model_types: Optional[List[str]] = None  # 複数モデル平均の場合
    log_level: str = "ERROR"
    save_intermediate_results: bool = True

    # フォールバック設定
    fallback_start_date: str = "2024-01-01"
    fallback_end_date: str = "2024-04-09"

    # MLフィルター設定
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

    def to_dict(self) -> Dict[str, Any]:
        """
        設定オブジェクトを辞書形式に変換

        DBへの保存やJSONシリアライズのために使用されます。

        Returns:
            設定値を含む辞書
        """
        return dataclass_to_dict(self)

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
        # デフォルト値を設定
        defaults = {
            "population_size": GA_DEFAULT_CONFIG["population_size"],
            "generations": GA_DEFAULT_CONFIG["generations"],
            "crossover_rate": GA_DEFAULT_CONFIG["crossover_rate"],
            "mutation_rate": GA_DEFAULT_CONFIG["mutation_rate"],
            "elite_size": GA_DEFAULT_CONFIG.get("elite_size", 10),
            "primary_metric": "sharpe_ratio",
            "fitness_weights": DEFAULT_FITNESS_WEIGHTS,
            "fitness_constraints": DEFAULT_FITNESS_CONSTRAINTS,
            "max_indicators": GA_DEFAULT_CONFIG["max_indicators"],
            "parameter_ranges": GA_PARAMETER_RANGES,
            "threshold_ranges": GA_THRESHOLD_RANGES,
            "min_indicators": 1,
            "min_conditions": 1,
            "max_conditions": 3,
            "zero_trades_penalty": GA_DEFAULT_CONFIG["zero_trades_penalty"],
            "constraint_violation_penalty": GA_DEFAULT_CONFIG[
                "constraint_violation_penalty"
            ],
            "enable_fitness_sharing": GA_DEFAULT_FITNESS_SHARING[
                "enable_fitness_sharing"
            ],
            "sharing_radius": GA_DEFAULT_FITNESS_SHARING["sharing_radius"],
            "sharing_alpha": GA_DEFAULT_FITNESS_SHARING["sharing_alpha"],
            "sampling_threshold": GA_DEFAULT_FITNESS_SHARING["sampling_threshold"],
            "sampling_ratio": GA_DEFAULT_FITNESS_SHARING["sampling_ratio"],
            "enable_multi_objective": False,
            "objectives": DEFAULT_GA_OBJECTIVES,
            "objective_weights": DEFAULT_GA_OBJECTIVE_WEIGHTS,
        }

        # デフォルト値をマージ
        for key, default_value in defaults.items():
            if key not in data or data[key] is None:
                data[key] = default_value

        # ネスト辞書からのサブ設定復元
        working = dict(data)
        if "mutation_config" in working and isinstance(working["mutation_config"], dict):
            working["mutation_config"] = MutationConfig.from_dict(working["mutation_config"])
        if "evaluation_config" in working and isinstance(working["evaluation_config"], dict):
            working["evaluation_config"] = EvaluationConfig.from_dict(working["evaluation_config"])
        if "hybrid_config" in working and isinstance(working["hybrid_config"], dict):
            working["hybrid_config"] = HybridConfig.from_dict(working["hybrid_config"])
        if "tuning_config" in working and isinstance(working["tuning_config"], dict):
            working["tuning_config"] = TuningConfig.from_dict(working["tuning_config"])

        # BaseConfigのfrom_dict処理を使用
        return cast(GAConfig, super().from_dict(working))

    def get_default_values(self) -> Dict[str, Any]:
        """BaseConfig用のデフォルト値を取得（自動生成を利用）"""
        # フィールドから自動生成したデフォルト値を取得
        defaults = self.get_default_values_from_fields()
        return {
            **defaults,
            "primary_metric": self.primary_metric,
        }
