"""
GASettingsクラス

遺伝的アルゴリズム設定を管理します。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .base import BaseConfig
from ..constants import (
    GA_DEFAULT_CONFIG,
    DEFAULT_FITNESS_WEIGHTS,
    DEFAULT_FITNESS_CONSTRAINTS,
    GA_PARAMETER_RANGES,
    GA_THRESHOLD_RANGES,
    GA_DEFAULT_FITNESS_SHARING,
    DEFAULT_GA_OBJECTIVES,
    DEFAULT_GA_OBJECTIVE_WEIGHTS,
)


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