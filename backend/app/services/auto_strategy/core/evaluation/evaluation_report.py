"""
評価レポートモデル

単一評価・OOS・WFA・PurgedKFold の評価結果を統一的に表現する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .. import objective_registry

_ROBUST_WORST_CASE_WEIGHT = 0.3


@dataclass(frozen=True)
class ScenarioEvaluation:
    """単一シナリオの評価結果。"""

    name: str
    fitness: Tuple[float, ...]
    passed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """複数シナリオの評価を束ねたレポート。"""

    mode: str
    objectives: Tuple[str, ...]
    aggregated_fitness: Tuple[float, ...]
    scenarios: List[ScenarioEvaluation] = field(default_factory=list)
    aggregate_method: str = "single"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        """通過シナリオ比率。"""
        if not self.scenarios:
            return 0.0
        passed_count = sum(1 for scenario in self.scenarios if scenario.passed)
        return passed_count / len(self.scenarios)

    @property
    def primary_objective(self) -> Optional[str]:
        """主要目的関数名。"""
        if not self.objectives:
            return None
        return self.objectives[0]

    @property
    def primary_aggregated_fitness(self) -> Optional[float]:
        """主要目的の集約 fitness。"""
        if not self.aggregated_fitness:
            return None
        return float(self.aggregated_fitness[0])

    @property
    def primary_worst_case_fitness(self) -> Optional[float]:
        """主要目的における最悪シナリオ値。"""
        if not self.scenarios or not self.objectives:
            return None

        values = [
            float(scenario.fitness[0])
            for scenario in self.scenarios
            if scenario.fitness
        ]
        if not values:
            return None

        if objective_registry.is_minimize_objective(self.objectives[0]):
            return float(max(values))
        return float(min(values))

    @classmethod
    def single(
        cls,
        *,
        mode: str,
        objectives: Sequence[str],
        scenario: ScenarioEvaluation,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "EvaluationReport":
        """単一シナリオのレポートを構築する。"""
        report = cls(
            mode=mode,
            objectives=tuple(objectives),
            aggregated_fitness=tuple(float(value) for value in scenario.fitness),
            scenarios=[scenario],
            aggregate_method="single",
            metadata=metadata.copy() if metadata else {},
        )
        report.metadata.setdefault("pass_rate", report.pass_rate)
        report.metadata.setdefault("scenario_count", 1)
        return report

    @classmethod
    def aggregate(
        cls,
        *,
        mode: str,
        objectives: Sequence[str],
        scenarios: Iterable[ScenarioEvaluation],
        aggregate_method: str = "robust",
        weights: Optional[Sequence[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "EvaluationReport":
        """複数シナリオを集約したレポートを構築する。"""
        scenario_list = list(scenarios)
        objective_names = tuple(objectives)

        if not scenario_list:
            return cls(
                mode=mode,
                objectives=objective_names,
                aggregated_fitness=tuple(),
                scenarios=[],
                aggregate_method=aggregate_method,
                metadata=metadata.copy() if metadata else {},
            )

        aggregated_fitness = []
        for index, objective in enumerate(objective_names):
            values = [
                float(scenario.fitness[index]) if index < len(scenario.fitness) else 0.0
                for scenario in scenario_list
            ]
            aggregated_fitness.append(
                cls._aggregate_objective_values(
                    values=values,
                    objective=objective,
                    aggregate_method=aggregate_method,
                    weights=weights,
                )
            )

        report = cls(
            mode=mode,
            objectives=objective_names,
            aggregated_fitness=tuple(aggregated_fitness),
            scenarios=scenario_list,
            aggregate_method=aggregate_method,
            metadata=metadata.copy() if metadata else {},
        )
        report.metadata.setdefault("pass_rate", report.pass_rate)
        report.metadata.setdefault("scenario_count", len(scenario_list))
        if weights is not None:
            report.metadata.setdefault("weights", list(weights))
        return report

    @classmethod
    def _aggregate_objective_values(
        cls,
        *,
        values: Sequence[float],
        objective: str,
        aggregate_method: str,
        weights: Optional[Sequence[float]],
    ) -> float:
        """目的関数ごとの値を集約する。"""
        if not values:
            return 0.0

        if aggregate_method == "single":
            return float(values[0])

        if aggregate_method == "weighted":
            normalized_weights = cls._normalize_weights(weights, len(values))
            return float(
                sum(value * weight for value, weight in zip(values, normalized_weights))
            )

        if aggregate_method == "mean":
            return float(sum(values) / len(values))

        if aggregate_method == "robust":
            center_value = float(median(values))
            worst_value = (
                max(values)
                if objective_registry.is_minimize_objective(objective)
                else min(values)
            )
            return float(
                center_value * (1.0 - _ROBUST_WORST_CASE_WEIGHT)
                + worst_value * _ROBUST_WORST_CASE_WEIGHT
            )

        raise ValueError(f"未対応の集約方式です: {aggregate_method}")

    @staticmethod
    def _normalize_weights(
        weights: Optional[Sequence[float]],
        expected_count: int,
    ) -> List[float]:
        """重み列を正規化する。"""
        if not weights or len(weights) != expected_count:
            return [1.0 / expected_count] * expected_count

        total = float(sum(weights))
        if total <= 0:
            return [1.0 / expected_count] * expected_count

        return [float(weight) / total for weight in weights]

    def to_dict(self) -> Dict[str, Any]:
        """保存・ログ出力向けに辞書へ変換する。"""
        return {
            "mode": self.mode,
            "objectives": list(self.objectives),
            "aggregated_fitness": list(self.aggregated_fitness),
            "aggregate_method": self.aggregate_method,
            "pass_rate": self.pass_rate,
            "metadata": self.metadata,
            "scenarios": [
                {
                    "name": scenario.name,
                    "fitness": list(scenario.fitness),
                    "passed": scenario.passed,
                    "metadata": scenario.metadata,
                    "performance_metrics": scenario.performance_metrics,
                }
                for scenario in self.scenarios
            ],
        }

    def to_summary_dict(self, max_scenarios: Optional[int] = None) -> Dict[str, Any]:
        """保存向けの軽量 summary を返す。"""
        if max_scenarios is None or max_scenarios < 0:
            scenario_slice = self.scenarios
        else:
            scenario_slice = self.scenarios[:max_scenarios]

        return {
            "mode": self.mode,
            "objectives": list(self.objectives),
            "aggregated_fitness": list(self.aggregated_fitness),
            "aggregate_method": self.aggregate_method,
            "pass_rate": self.pass_rate,
            "scenario_count": len(self.scenarios),
            "truncated_scenario_count": max(
                0, len(self.scenarios) - len(scenario_slice)
            ),
            "primary_objective": self.primary_objective,
            "primary_aggregated_fitness": self.primary_aggregated_fitness,
            "primary_worst_case_fitness": self.primary_worst_case_fitness,
            "metadata": self.metadata.copy(),
            "scenarios": [
                {
                    "name": scenario.name,
                    "passed": scenario.passed,
                    "fitness": list(scenario.fitness),
                    "primary_fitness": (
                        float(scenario.fitness[0]) if scenario.fitness else 0.0
                    ),
                    "metadata": scenario.metadata.copy(),
                }
                for scenario in scenario_slice
            ],
        }
