"""
GA実行時設定クラス

GAConfigクラスとGAProgressクラスを提供します。
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, cast

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

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "population_size": self.population_size,
            "generations": self.generations,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "elite_size": self.elite_size,
            "fitness_weights": self.fitness_weights,
            "primary_metric": self.primary_metric,
            "max_indicators": self.max_indicators,
            "enable_multi_objective": self.enable_multi_objective,
            "objectives": self.objectives,
            "objective_weights": self.objective_weights,
            "dynamic_objective_reweighting": self.dynamic_objective_reweighting,
            "objective_dynamic_scalars": self.objective_dynamic_scalars,
            "parameter_ranges": self.parameter_ranges,
            "threshold_ranges": self.threshold_ranges,
            "fitness_constraints": self.fitness_constraints,
            "min_indicators": self.min_indicators,
            "min_conditions": self.min_conditions,
            "max_conditions": self.max_conditions,
            "zero_trades_penalty": self.zero_trades_penalty,
            "constraint_violation_penalty": self.constraint_violation_penalty,
            "crossover_field_selection_probability": self.crossover_field_selection_probability,
            "indicator_param_mutation_range": self.indicator_param_mutation_range,
            "risk_param_mutation_range": self.risk_param_mutation_range,
            "indicator_add_delete_probability": self.indicator_add_delete_probability,
            "indicator_add_vs_delete_probability": self.indicator_add_vs_delete_probability,
            "condition_change_probability_multiplier": self.condition_change_probability_multiplier,
            "condition_selection_probability": self.condition_selection_probability,
            "condition_operator_switch_probability": self.condition_operator_switch_probability,
            "valid_condition_operators": self.valid_condition_operators,
            "tpsl_gene_creation_probability_multiplier": self.tpsl_gene_creation_probability_multiplier,
            "position_sizing_gene_creation_probability_multiplier": self.position_sizing_gene_creation_probability_multiplier,
            "adaptive_mutation_variance_threshold": self.adaptive_mutation_variance_threshold,
            "adaptive_mutation_rate_decrease_multiplier": self.adaptive_mutation_rate_decrease_multiplier,
            "adaptive_mutation_rate_increase_multiplier": self.adaptive_mutation_rate_increase_multiplier,
            "parallel_processes": self.parallel_processes,
            "random_state": self.random_state,
            "enable_parallel_evaluation": self.enable_parallel_evaluation,
            "max_evaluation_workers": self.max_evaluation_workers,
            "evaluation_timeout": self.evaluation_timeout,
            "log_level": self.log_level,
            "save_intermediate_results": self.save_intermediate_results,
            "ml_filter_enabled": self.ml_filter_enabled,
            "ml_model_path": self.ml_model_path,
            "preprocess_features": self.preprocess_features,
            "fallback_start_date": self.fallback_start_date,
            "fallback_end_date": self.fallback_end_date,
            "enable_fitness_sharing": self.enable_fitness_sharing,
            "sharing_radius": self.sharing_radius,
            "sharing_alpha": self.sharing_alpha,
            "sampling_threshold": self.sampling_threshold,
            "sampling_ratio": self.sampling_ratio,
            "tpsl_method_constraints": self.tpsl_method_constraints,
            "tpsl_sl_range": self.tpsl_sl_range,
            "tpsl_tp_range": self.tpsl_tp_range,
            "tpsl_rr_range": self.tpsl_rr_range,
            "tpsl_atr_multiplier_range": self.tpsl_atr_multiplier_range,
            "oos_split_ratio": self.oos_split_ratio,
            "oos_fitness_weight": self.oos_fitness_weight,
            "enable_walk_forward": self.enable_walk_forward,
            "wfa_n_folds": self.wfa_n_folds,
            "wfa_train_ratio": self.wfa_train_ratio,
            "wfa_anchored": self.wfa_anchored,
            "enable_multi_timeframe": self.enable_multi_timeframe,
            "available_timeframes": self.available_timeframes,
            "mtf_indicator_probability": self.mtf_indicator_probability,
            "parameter_range_preset": self.parameter_range_preset,
            "enable_parameter_tuning": self.enable_parameter_tuning,
            "tuning_n_trials": self.tuning_n_trials,
            "tuning_elite_count": self.tuning_elite_count,
            "tuning_use_wfa": self.tuning_use_wfa,
            "tuning_include_indicators": self.tuning_include_indicators,
            "tuning_include_tpsl": self.tuning_include_tpsl,
            "tuning_include_thresholds": self.tuning_include_thresholds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GAConfig":
        """辞書から復元"""
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

        # BaseConfigのfrom_dict処理を使用
        return cast(GAConfig, super().from_dict(data))

    def get_default_values(self) -> Dict[str, Any]:
        """BaseConfig用のデフォルト値を取得（自動生成を利用）"""
        # フィールドから自動生成したデフォルト値を取得
        defaults = self.get_default_values_from_fields()
        return {
            **defaults,
            "primary_metric": self.primary_metric,
        }

    def to_json(self) -> str:
        """JSON文字列に変換（BaseConfigの機能を活用）"""
        return super().to_json()

    @classmethod
    def from_json(cls, json_str: str) -> "GAConfig":
        """JSON文字列から復元（BaseConfigの機能を活用）"""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"JSON復元エラー: {e}", exc_info=True)
            raise ValueError(f"JSON からの復元に失敗しました: {e}")
