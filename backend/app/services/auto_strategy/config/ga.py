"""
GASettingsクラス

遺伝的アルゴリズム設定を管理します。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .base import BaseConfig

# GA基本設定
GA_DEFAULT_CONFIG = {
    "population_size": 100,  # より実用的、デフォルトをGA_DEFAULT_SETTINGSに合わせる
    "generations": 50,
    "crossover_rate": 0.8,
    "mutation_rate": 0.1,
    "elite_size": 10,
    "max_indicators": 3,
}

# フィットネス重み設定
FITNESS_WEIGHT_PROFILES = {
    "balanced": {
        "total_return": 0.25,
        "sharpe_ratio": 0.35,
        "max_drawdown": 0.2,
        "win_rate": 0.1,
        "balance_score": 0.1,
    },
}

DEFAULT_FITNESS_WEIGHTS = FITNESS_WEIGHT_PROFILES["balanced"]

# フィットネス制約設定
DEFAULT_FITNESS_CONSTRAINTS = {
    "min_trades": 10,
    "max_drawdown_limit": 0.3,
    "min_sharpe_ratio": 1.0,
}

# GA目的設定
DEFAULT_GA_OBJECTIVES = ["total_return"]
DEFAULT_GA_OBJECTIVE_WEIGHTS = [1.0]  # 最大化

# GAフィットネス共有設定
GA_DEFAULT_FITNESS_SHARING = {
    "enable_fitness_sharing": True,
    "sharing_radius": 0.1,
    "sharing_alpha": 1.0,
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


@dataclass
class GASettings(BaseConfig):
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

    def get_default_values(self) -> Dict[str, Any]:
        """デフォルト値を取得（自動生成を利用）"""
        # フィールドから自動生成したデフォルト値を取得
        defaults = self.get_default_values_from_fields()
        # 必要に応じてカスタマイズ（外部定数など）
        return defaults

    def _custom_validation(self) -> List[str]:
        """カスタム検証"""
        errors = []

        if self.population_size <= 0:
            errors.append("人口サイズは正の整数である必要があります")

        if not (0 <= self.crossover_rate <= 1):
            errors.append("交叉率は0から1の範囲である必要があります")

        if not (0 <= self.mutation_rate <= 1):
            errors.append("突然変異率は0から1の範囲である必要があります")

        if self.elite_size >= self.population_size:
            errors.append("エリートサイズは人口サイズより小さく設定してください")

        if self.min_indicators > self.max_indicators:
            errors.append("最小指標数は最大指標数以下である必要があります")

        return errors

    def __post_init__(self) -> None:
        """Post-initialization validation"""
        # Validate integer fields
        if not isinstance(self.population_size, int) or self.population_size <= 0:
            raise ValueError("population_size は正の整数である必要があります")
        if not isinstance(self.generations, int) or self.generations <= 0:
            raise ValueError("generations は正の整数である必要があります")
        if not isinstance(self.elite_size, int) or self.elite_size < 0:
            raise ValueError("elite_size は負でない整数である必要があります")
        if not isinstance(self.max_indicators, int) or self.max_indicators <= 0:
            raise ValueError("max_indicators は正の整数である必要があります")

        # Validate float fields
        if not isinstance(self.crossover_rate, (int, float)) or not (0 <= self.crossover_rate <= 1):
            raise ValueError("crossover_rate は0から1の範囲の実数である必要があります")
        if not isinstance(self.mutation_rate, (int, float)) or not (0 <= self.mutation_rate <= 1):
            raise ValueError("mutation_rate は0から1の範囲の実数である必要があります")

        # Convert int to float if necessary
        if isinstance(self.crossover_rate, int):
            self.crossover_rate = float(self.crossover_rate)
        if isinstance(self.mutation_rate, int):
            self.mutation_rate = float(self.mutation_rate)