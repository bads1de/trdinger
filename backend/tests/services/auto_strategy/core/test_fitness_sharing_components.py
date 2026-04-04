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
from app.services.auto_strategy.core.fitness.fitness_sharing_vectorizer import (
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
        assert len(vector) == 7 + len(indicator_types) + len(operator_types) + 4

        operator_start = 7 + len(indicator_types)
        assert vector[operator_start + operator_map["AND"]] == 1.0
        assert vector[operator_start + operator_map["OR"]] == 1.0
        assert vector[-2] == pytest.approx(2.0)
        assert vector[-1] == pytest.approx(1.0)

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
