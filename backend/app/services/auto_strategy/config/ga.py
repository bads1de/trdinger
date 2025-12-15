"""
GASettingsクラス

遺伝的アルゴリズム設定を管理します。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseConfig

# GA基本設定
GA_DEFAULT_CONFIG = {
    "population_size": 100,  # より実用的、デフォルトをGA_DEFAULT_SETTINGSに合わせる
    "generations": 50,
    "crossover_rate": 0.8,
    "mutation_rate": 0.1,
    "elite_size": 10,
    "max_indicators": 3,
    "zero_trades_penalty": 0.1,  # 取引回数0回のペナルティスコア
    "constraint_violation_penalty": 0.0,  # 制約違反時のスコア
}

# フィットネス重み設定
FITNESS_WEIGHT_PROFILES = {
    "balanced": {
        "total_return": 0.2,
        "sharpe_ratio": 0.25,
        "max_drawdown": 0.15,
        "win_rate": 0.1,
        "balance_score": 0.1,
        "ulcer_index_penalty": 0.15,
        "trade_frequency_penalty": 0.05,
    },
}

DEFAULT_FITNESS_WEIGHTS = FITNESS_WEIGHT_PROFILES["balanced"]

# フィットネス制約設定
DEFAULT_FITNESS_CONSTRAINTS = {
    "min_trades": 10,
    "max_drawdown_limit": 0.3,
    "min_sharpe_ratio": 1.0,
}

# GA目的設定（デフォルトはweighted_score: 従来の重み付け単一目的）
DEFAULT_GA_OBJECTIVES = [
    "weighted_score",
]
DEFAULT_GA_OBJECTIVE_WEIGHTS = [1.0]

# GAフィットネス共有設定
GA_DEFAULT_FITNESS_SHARING = {
    "enable_fitness_sharing": True,
    "sharing_radius": 0.1,
    "sharing_alpha": 1.0,
    "sampling_threshold": 200,
    "sampling_ratio": 0.3,
}

# GAパラメータ範囲定義
GA_PARAMETER_RANGES = {
    # 基本パラメータ
    "period": [5, 200],
    "fast_period": [5, 20],
    "slow_period": [20, 50],
    "signal_period": [5, 15],
    # 特殊パラメータ
    "std_dev": [1.5, 2.5],
    "k_period": [10, 20],
    "d_period": [3, 7],
    "slowing": [1, 5],
    # 閾値パラメータ
    "overbought": [70, 90],
    "oversold": [10, 30],
}

# GA閾値範囲定義
GA_THRESHOLD_RANGES = {
    "oscillator_0_100": [20, 80],
    "oscillator_plus_minus_100": [-100, 100],
    "momentum_zero_centered": [-0.5, 0.5],
    "funding_rate": [0.0001, 0.0005, 0.001, -0.0001, -0.0005, -0.001],
    "open_interest": [1000000, 5000000, 10000000, 50000000],
    "price_ratio": [0.95, 1.05],
}


GA_MUTATION_SETTINGS = {
    "indicator_param_mutation_range": (
        0.8,
        1.2,
    ),  # 指標パラメータの変動幅 (min_multiplier, max_multiplier)
    "indicator_add_delete_probability": 0.3,  # 指標の追加・削除確率 (x mutation_rate)
    "indicator_add_vs_delete_probability": 0.5,  # 指標追加 vs 削除の確率 (閾値)
    "crossover_field_selection_probability": 0.5,  # ユニフォーム交叉で親1のフィールドを選択する確率
    "condition_operator_switch_probability": 0.5,  # 条件グループのAND/OR切り替え確率
    "condition_change_probability_multiplier": 0.5,  # 個別条件の変更確率 (x mutation_rate)
    "condition_selection_probability": 0.5,  # エントリー/エグジット条件選択確率
    "risk_param_mutation_range": (0.8, 1.2),  # リスク管理パラメータの変動幅
    "tpsl_gene_creation_probability_multiplier": 0.2,  # 欠損TP/SL遺伝子の新規作成確率 (x mutation_rate)
    "position_sizing_gene_creation_probability_multiplier": 0.2,  # 欠損PS遺伝子の新規作成確率 (x mutation_rate)
    "adaptive_mutation_variance_threshold": 0.1,  # 適応的突然変異の分散閾値
    "adaptive_mutation_rate_decrease_multiplier": 0.5,  # 適応的突然変異のレート減少倍率
    "adaptive_mutation_rate_increase_multiplier": 2.0,  # 適応的突然変異のレート増加倍率
    "valid_condition_operators": [">", "<", ">=", "<=", "=="],  # 有効な条件演算子
    "numeric_threshold_probability": 0.3,  # 数値閾値を生成する確率
    "min_compatibility_score": 0.5,  # オペランド互換性の最低スコア
    "strict_compatibility_score": 0.7,  # オペランド互換性の厳密スコア
}


@dataclass
class GASettings(BaseConfig):
    # 後方互換性のため
    GAConfig = None
    """遺伝的アルゴリズム設定"""

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

    # 突然変異設定
    indicator_param_mutation_range: Tuple[float, float] = GA_MUTATION_SETTINGS[
        "indicator_param_mutation_range"
    ]
    indicator_add_delete_probability: float = GA_MUTATION_SETTINGS[
        "indicator_add_delete_probability"
    ]
    indicator_add_vs_delete_probability: float = GA_MUTATION_SETTINGS[
        "indicator_add_vs_delete_probability"
    ]
    crossover_field_selection_probability: float = GA_MUTATION_SETTINGS[
        "crossover_field_selection_probability"
    ]
    condition_operator_switch_probability: float = GA_MUTATION_SETTINGS[
        "condition_operator_switch_probability"
    ]
    condition_change_probability_multiplier: float = GA_MUTATION_SETTINGS[
        "condition_change_probability_multiplier"
    ]
    condition_selection_probability: float = GA_MUTATION_SETTINGS[
        "condition_selection_probability"
    ]
    risk_param_mutation_range: Tuple[float, float] = GA_MUTATION_SETTINGS[
        "risk_param_mutation_range"
    ]
    tpsl_gene_creation_probability_multiplier: float = GA_MUTATION_SETTINGS[
        "tpsl_gene_creation_probability_multiplier"
    ]
    position_sizing_gene_creation_probability_multiplier: float = GA_MUTATION_SETTINGS[
        "position_sizing_gene_creation_probability_multiplier"
    ]
    adaptive_mutation_variance_threshold: float = GA_MUTATION_SETTINGS[
        "adaptive_mutation_variance_threshold"
    ]
    adaptive_mutation_rate_decrease_multiplier: float = GA_MUTATION_SETTINGS[
        "adaptive_mutation_rate_decrease_multiplier"
    ]
    adaptive_mutation_rate_increase_multiplier: float = GA_MUTATION_SETTINGS[
        "adaptive_mutation_rate_increase_multiplier"
    ]
    valid_condition_operators: List[str] = field(
        default_factory=lambda: GA_MUTATION_SETTINGS["valid_condition_operators"].copy()
    )
    numeric_threshold_probability: float = GA_MUTATION_SETTINGS[
        "numeric_threshold_probability"
    ]
    min_compatibility_score: float = GA_MUTATION_SETTINGS["min_compatibility_score"]
    strict_compatibility_score: float = GA_MUTATION_SETTINGS[
        "strict_compatibility_score"
    ]

    # マルチタイムフレーム設定
    enable_multi_timeframe: bool = False
    mtf_indicator_probability: float = 0.3
    available_timeframes: List[str] = field(default_factory=list)

    # パラメータ範囲
    parameter_ranges: Dict[str, List] = field(
        default_factory=lambda: GA_PARAMETER_RANGES.copy()
    )
    threshold_ranges: Dict[str, List[float]] = field(
        default_factory=lambda: GA_THRESHOLD_RANGES.copy()
    )

    # フィットネス設定
    fitness_weights: Dict[str, float] = field(
        default_factory=lambda: FITNESS_WEIGHT_PROFILES["balanced"].copy()
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
    ga_objectives: List[str] = field(
        default_factory=lambda: DEFAULT_GA_OBJECTIVES.copy()
    )
    ga_objective_weights: List[float] = field(
        default_factory=lambda: DEFAULT_GA_OBJECTIVE_WEIGHTS.copy()
    )

    # MLフィルター設定
    ml_filter_enabled: bool = False
    ml_model_path: Optional[str] = None

    # 階層的GA設定（サブGA）
    hierarchical_ga_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "population_size": 20,
            "generations": 10,
            "crossover_rate": 0.8,
            "mutation_rate": 0.2,
        }
    )

    def get_default_values(self) -> Dict[str, Any]:
        """デフォルト値を取得（自動生成を利用）"""
        # フィールドから自動生成したデフォルト値を取得
        defaults = self.get_default_values_from_fields()
        # 必要に応じてカスタマイズ（外部定数など）
        return defaults





