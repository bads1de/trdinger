import copy
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from app.services.auto_strategy.core.fitness.fitness_sharing import FitnessSharing
from app.services.auto_strategy.core.fitness.fitness_sharing_niche import (
    compute_niche_counts_vectorized,
)
from app.services.auto_strategy.core.fitness.fitness_sharing_similarity import (
    calculate_similarity,
)
from app.services.auto_strategy.core.fitness import fitness_sharing_silhouette
from app.services.auto_strategy.core.fitness.fitness_sharing_vectorizer import (
    BEHAVIOR_FEATURE_NAMES,
    build_behavior_profile,
    vectorize_gene,
)
from app.services.auto_strategy.genes import (
    Condition,
    ConditionGroup,
    IndicatorGene,
    PositionSizingGene,
    StrategyGene,
    TPSLGene,
)


class TestFitnessSharingComponents:
    def test_vectorize_gene_counts_nested_groups_and_operand_types(self) -> None:
        gene = StrategyGene(
            id="gene",
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 10}),
                IndicatorGene(type="EMA", parameters={"period": 20}),
            ],
            long_entry_conditions=[
                ConditionGroup(
                    operator="AND",
                    conditions=[
                        Condition(
                            left_operand="close",
                            operator=">",
                            right_operand="sma",
                        ),
                        ConditionGroup(
                            operator="OR",
                            conditions=[
                                Condition(
                                    left_operand="rsi",
                                    operator="<",
                                    right_operand="30",
                                ),
                                Condition(
                                    left_operand="adx",
                                    operator=">",
                                    right_operand="25",
                                ),
                            ],
                        ),
                    ],
                )
            ],
            short_entry_conditions=[],
            risk_management={"position_size": 0.15},
            tpsl_gene=TPSLGene(stop_loss_pct=0.03, take_profit_pct=0.08),
            position_sizing_gene=PositionSizingGene(risk_per_trade=0.02),
            metadata={},
        )

        indicator_types = ["EMA", "RSI", "SMA"]
        indicator_map = {name: i for i, name in enumerate(indicator_types)}
        operator_types = ["<", ">", "AND", "OR"]
        operator_map = {name: i for i, name in enumerate(operator_types)}

        vector = vectorize_gene(
            gene,
            indicator_types=indicator_types,
            indicator_map=indicator_map,
            operator_types=operator_types,
            operator_map=operator_map,
        )

        assert isinstance(vector, np.ndarray)
        assert len(vector) == (
            7
            + len(indicator_types)
            + len(operator_types)
            + 4
            + len(BEHAVIOR_FEATURE_NAMES)
        )

        operator_start = 7 + len(indicator_types)
        assert vector[operator_start + operator_map["AND"]] == 1.0
        assert vector[operator_start + operator_map["OR"]] == 1.0
        behavior_offset = len(BEHAVIOR_FEATURE_NAMES)
        assert vector[-(behavior_offset + 2)] == pytest.approx(2.0)
        assert vector[-(behavior_offset + 1)] == pytest.approx(1.0)

    def test_build_behavior_profile_uses_report_metrics(self) -> None:
        report = SimpleNamespace(
            pass_rate=0.5,
            primary_aggregated_fitness=1.25,
            primary_worst_case_fitness=0.75,
            scenarios=[
                SimpleNamespace(
                    performance_metrics={
                        "total_return": 0.12,
                        "sharpe_ratio": 1.4,
                        "max_drawdown": 0.08,
                        "total_trades": 18,
                    }
                ),
                SimpleNamespace(
                    performance_metrics={
                        "total_return": 0.06,
                        "sharpe_ratio": 0.8,
                        "max_drawdown": 0.15,
                        "total_trades": 10,
                    }
                ),
            ],
        )

        profile = build_behavior_profile(
            fitness_values=(1.0, 0.6),
            evaluation_report=report,
        )

        assert profile["objective_count"] == pytest.approx(2.0)
        assert profile["pass_rate"] == pytest.approx(0.5)
        assert profile["scenario_count"] == pytest.approx(2.0)
        assert profile["aggregated_primary"] == pytest.approx(1.25)
        assert profile["worst_case_primary"] == pytest.approx(0.75)
        assert profile["mean_total_return"] == pytest.approx(0.09)
        assert profile["mean_total_trades"] == pytest.approx(14.0)

    def test_compute_niche_counts_sampling_preserves_global_rng_state(self) -> None:
        np.random.seed(123)
        vectors = np.random.rand(12, 4)
        state_before = cast(tuple[Any, Any, Any, Any, Any], np.random.get_state())

        counts = compute_niche_counts_vectorized(
            vectors,
            sharing_radius=0.1,
            sampling_threshold=5,
            sampling_ratio=0.5,
        )

        state_after = cast(tuple[Any, Any, Any, Any, Any], np.random.get_state())

        assert counts.shape == (12,)
        assert np.all(counts >= 1.0)
        assert state_before[0] == state_after[0]
        assert np.array_equal(state_before[1], state_after[1])
        assert state_before[2] == state_after[2]
        assert state_before[3] == state_after[3]
        assert state_before[4] == state_after[4]

    def test_calculate_similarity_identical_genes_returns_one(self) -> None:
        gene = StrategyGene(
            id="gene",
            indicators=[IndicatorGene(type="SMA", parameters={"period": 10})],
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="sma")
            ],
            short_entry_conditions=[],
            risk_management={"position_size": 0.1},
            tpsl_gene=TPSLGene(stop_loss_pct=0.03, take_profit_pct=0.08),
            position_sizing_gene=PositionSizingGene(risk_per_trade=0.02),
            metadata={},
        )

        identical_gene = copy.deepcopy(gene)

        assert calculate_similarity(gene, identical_gene) == pytest.approx(1.0)

    def test_silhouette_based_sharing_returns_early_for_two_individuals(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called = {"kmeans": False}

        class FakeKMeans:
            def __init__(self, n_clusters: int, random_state: int, n_init: Any):
                called["kmeans"] = True

            def fit_predict(self, vectors: np.ndarray) -> np.ndarray:
                return np.array([0, 1])

        monkeypatch.setattr(fitness_sharing_silhouette, "KMeans", FakeKMeans)
        monkeypatch.setattr(
            fitness_sharing_silhouette,
            "silhouette_samples",
            lambda vectors, labels: np.array([0.1, 0.2]),
        )

        sharing = FitnessSharing(sharing_radius=0.1, alpha=1.0)
        population = [
            StrategyGene(
                id="gene1",
                indicators=[IndicatorGene(type="SMA", parameters={"period": 10})],
                long_entry_conditions=[],
                short_entry_conditions=[],
                risk_management={},
                tpsl_gene=None,
                position_sizing_gene=None,
                metadata={},
            ),
            StrategyGene(
                id="gene2",
                indicators=[IndicatorGene(type="EMA", parameters={"period": 20})],
                long_entry_conditions=[],
                short_entry_conditions=[],
                risk_management={},
                tpsl_gene=None,
                position_sizing_gene=None,
                metadata={},
            ),
        ]

        result = sharing.silhouette_based_sharing(population)

        assert result == population
        assert called["kmeans"] is False

    def test_silhouette_based_sharing_caps_clusters_for_three_individuals(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, int] = {}

        class FakeKMeans:
            def __init__(self, n_clusters: int, random_state: int, n_init: Any):
                captured["n_clusters"] = n_clusters

            def fit_predict(self, vectors: np.ndarray) -> np.ndarray:
                return np.array([0, 0, 1])

        monkeypatch.setattr(fitness_sharing_silhouette, "KMeans", FakeKMeans)
        monkeypatch.setattr(
            fitness_sharing_silhouette,
            "silhouette_samples",
            lambda vectors, labels: np.array([0.1, 0.2, 0.3]),
        )

        sharing = FitnessSharing(sharing_radius=0.1, alpha=1.0)
        population = [
            StrategyGene(
                id="gene1",
                indicators=[IndicatorGene(type="SMA", parameters={"period": 10})],
                long_entry_conditions=[],
                short_entry_conditions=[],
                risk_management={},
                tpsl_gene=None,
                position_sizing_gene=None,
                metadata={},
            ),
            StrategyGene(
                id="gene2",
                indicators=[IndicatorGene(type="EMA", parameters={"period": 20})],
                long_entry_conditions=[],
                short_entry_conditions=[],
                risk_management={},
                tpsl_gene=None,
                position_sizing_gene=None,
                metadata={},
            ),
            StrategyGene(
                id="gene3",
                indicators=[IndicatorGene(type="RSI", parameters={"period": 14})],
                long_entry_conditions=[],
                short_entry_conditions=[],
                risk_management={},
                tpsl_gene=None,
                position_sizing_gene=None,
                metadata={},
            ),
        ]

        result = sharing.silhouette_based_sharing(population)

        assert result == population
        assert captured["n_clusters"] == 2

    def test_apply_fitness_sharing_does_not_attach_feature_vector(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FitnessAwareStrategyGene(StrategyGene):
            __slots__ = ("fitness",)

        gene1 = FitnessAwareStrategyGene(
            id="gene1",
            indicators=[IndicatorGene(type="SMA", parameters={"period": 10})],
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="sma")
            ],
            short_entry_conditions=[],
            risk_management={"position_size": 0.1},
            tpsl_gene=TPSLGene(),
            position_sizing_gene=PositionSizingGene(),
            metadata={},
        )
        gene2 = FitnessAwareStrategyGene(
            id="gene2",
            indicators=[IndicatorGene(type="EMA", parameters={"period": 20})],
            long_entry_conditions=[
                Condition(left_operand="close", operator="<", right_operand="ema")
            ],
            short_entry_conditions=[],
            risk_management={"position_size": 0.2},
            tpsl_gene=TPSLGene(),
            position_sizing_gene=PositionSizingGene(),
            metadata={},
        )
        gene1.fitness = SimpleNamespace(values=(1.0,), valid=True)
        gene2.fitness = SimpleNamespace(values=(1.0,), valid=True)

        sharing = FitnessSharing(sharing_radius=0.1, alpha=1.0)
        monkeypatch.setattr(
            sharing,
            "compute_niche_counts_vectorized",
            lambda vectors: np.ones(len(vectors)),
        )
        monkeypatch.setattr(
            sharing,
            "silhouette_based_sharing",
            lambda population: population,
        )

        result = sharing.apply_fitness_sharing([gene1, gene2])

        assert len(result) == 2
        assert not hasattr(gene1, "_feature_vector")
        assert not hasattr(gene2, "_feature_vector")
