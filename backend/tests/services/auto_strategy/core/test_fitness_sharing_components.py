from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from app.services.auto_strategy.core.fitness import (
    fitness_sharing,
    fitness_sharing_silhouette,
)
from app.services.auto_strategy.core.fitness.fitness_sharing import FitnessSharing
from app.services.auto_strategy.core.fitness.fitness_sharing_niche import (
    compute_niche_counts_vectorized,
)
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

    def test_build_population_feature_vectors_returns_normalized_vectors(self) -> None:
        sharing = FitnessSharing(sharing_radius=0.1, alpha=1.0)
        sharing.set_evaluation_report_provider(
            lambda individual: (
                {"pass_rate": 0.2} if individual.id == "gene1" else {"pass_rate": 0.8}
            )
        )

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
                indicators=[IndicatorGene(type="EMA", parameters={"period": 30})],
                long_entry_conditions=[],
                short_entry_conditions=[],
                risk_management={},
                tpsl_gene=None,
                position_sizing_gene=None,
                metadata={},
            ),
        ]

        vectors = sharing.build_population_feature_vectors(population)

        assert set(vectors) == {id(population[0]), id(population[1])}
        assert vectors[id(population[0])].shape == vectors[id(population[1])].shape
        assert not np.array_equal(
            vectors[id(population[0])], vectors[id(population[1])]
        )
        assert np.all(vectors[id(population[0])] >= 0.0)
        assert np.all(vectors[id(population[0])] <= 1.0)

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

    def test_niche_threshold_ignores_dead_dimensions(self) -> None:
        """値が一定の次元（未使用指標など）がしきい値を広げないこと"""
        # 実効1次元: 正規化後の距離は 0.4 と 0.6
        active_values = np.array([[0.0], [0.4], [1.0]])
        dead_dimensions = np.zeros((3, 49))
        vectors_with_dead_dims = np.hstack([active_values, dead_dimensions])

        counts = compute_niche_counts_vectorized(
            vectors_with_dead_dims,
            sharing_radius=0.1,
            sampling_threshold=1000,
            sampling_ratio=0.3,
        )

        # 実効1次元なら半径は 0.1*sqrt(1)=0.1 → 距離0.4は隣人にならない
        np.testing.assert_allclose(counts, [1.0, 1.0, 1.0])

    def test_niche_threshold_finds_neighbors_in_active_dimensions(self) -> None:
        """実効次元内の近接個体は隣人として数えられること"""
        active_values = np.array([[0.0], [0.05], [1.0]])

        counts = compute_niche_counts_vectorized(
            active_values,
            sharing_radius=0.1,
            sampling_threshold=1000,
            sampling_ratio=0.3,
        )

        # 正規化後の距離 0.005 < 0.1 → 先頭2個体は互いに隣人（自分含む）
        np.testing.assert_allclose(counts, [2.0, 2.0, 1.0])

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

    def test_silhouette_adjustment_is_bounded_to_half_of_raw(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """シルエット倍率は生値の50%未満に下げないこと（順位情報の消滅防止）"""

        class FakeKMeans:
            def __init__(self, n_clusters: int, random_state: int, n_init: Any) -> None:
                pass

            def fit_predict(self, vectors: np.ndarray) -> np.ndarray:
                return np.array([0, 0, 1])

        monkeypatch.setattr(fitness_sharing_silhouette, "KMeans", FakeKMeans)
        # normalized silhouette は [1.0, 0.5, 0.0] → 倍率 [0.5, 0.75, 1.0]
        monkeypatch.setattr(
            fitness_sharing_silhouette,
            "silhouette_samples",
            lambda vectors, labels: np.array([1.0, 0.0, -1.0]),
        )

        sharing = FitnessSharing(sharing_radius=0.1, alpha=1.0)

        class FitnessAwareStrategyGene(StrategyGene):
            __slots__ = ("fitness",)

        population = []
        # 最悪値 1.0 からの優位幅が [0.0, 0.5, 1.0] になるように設定
        for i in range(3):
            gene = FitnessAwareStrategyGene(
                id=f"gene{i}",
                indicators=[IndicatorGene(type="SMA", parameters={"period": 10 + i})],
                long_entry_conditions=[],
                short_entry_conditions=[],
                risk_management={},
                tpsl_gene=None,
                position_sizing_gene=None,
                metadata={},
            )
            gene.fitness = SimpleNamespace(values=(1.0 + 0.5 * i,), valid=True)
            population.append(gene)

        result = sharing.silhouette_based_sharing(population)

        adjusted = [gene.fitness.values[0] for gene in result]
        # 優位幅に倍率 [0.5, 0.75, 1.0] を掛ける:
        # [1.0 + 0*0.5, 1.0 + 0.5*0.75, 1.0 + 1.0*1.0]
        np.testing.assert_allclose(adjusted, [1.0, 1.375, 2.0])

    def test_silhouette_sharing_never_improves_minimize_objective(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """最小化目的でシルエット調整が「改善」に働かないこと（方向バグ回帰）"""

        class FakeKMeans:
            def __init__(self, n_clusters: int, random_state: int, n_init: Any) -> None:
                pass

            def fit_predict(self, vectors: np.ndarray) -> np.ndarray:
                return np.array([0, 0, 1])

        monkeypatch.setattr(fitness_sharing_silhouette, "KMeans", FakeKMeans)
        monkeypatch.setattr(
            fitness_sharing_silhouette,
            "silhouette_samples",
            lambda vectors, labels: np.array([1.0, 1.0, 1.0]),
        )

        sharing = FitnessSharing(sharing_radius=0.1, alpha=1.0)

        class FitnessAwareStrategyGene(StrategyGene):
            __slots__ = ("fitness",)

        population = []
        # ドローダウン（最小化）: 小さいほど良い。全員クラスタ中心扱い。
        for i, drawdown in enumerate([0.30, 0.20, 0.10]):
            gene = FitnessAwareStrategyGene(
                id=f"gene{i}",
                indicators=[IndicatorGene(type="SMA", parameters={"period": 10 + i})],
                long_entry_conditions=[],
                short_entry_conditions=[],
                risk_management={},
                tpsl_gene=None,
                position_sizing_gene=None,
                metadata={},
            )
            gene.fitness = SimpleNamespace(values=(drawdown,), valid=True)
            population.append(gene)

        result = sharing.silhouette_based_sharing(
            population, objectives=["max_drawdown"]
        )

        adjusted = [gene.fitness.values[0] for gene in result]
        # 最悪（0.30）以外は調整後に必ず悪化（値が増加）する。旧実装の
        # 生値への乗算は最小化目的で値を 0 へ寄せて「改善」させていた。
        assert adjusted[0] == 0.30
        assert adjusted[1] > 0.20
        assert adjusted[2] > 0.10

    def test_apply_fitness_sharing_softens_niche_division(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """同一構造クラスタの優位幅は sqrt(ニッチ数) で縮小されること

        behavior profile は適応度値を含むため、適応度が異なる個体は
        自動的に別ニッチになる。ニッチ段のメカニクスを検証するため
        profile を空にして構造ベクトルのみで比較する。
        """
        sharing = FitnessSharing(sharing_radius=0.1, alpha=1.0)
        monkeypatch.setattr(
            sharing, "_build_behavior_profile_map", lambda population: {}
        )
        monkeypatch.setattr(
            fitness_sharing,
            "_silhouette_based_sharing",
            lambda population, gene_serializer, vectorize_gene, objectives=None: (
                population
            ),
        )

        class FitnessAwareStrategyGene(StrategyGene):
            __slots__ = ("fitness",)

        population = []
        # 同一構造（ニッチ数4）で適応度のみ異なる集団
        for i in range(4):
            gene = FitnessAwareStrategyGene(
                id=f"gene{i}",
                indicators=[IndicatorGene(type="SMA", parameters={"period": 20})],
                long_entry_conditions=[],
                short_entry_conditions=[],
                risk_management={},
                tpsl_gene=None,
                position_sizing_gene=None,
                metadata={},
            )
            gene.fitness = SimpleNamespace(values=(1.0 + i,), valid=True)
            population.append(gene)

        result = sharing.apply_fitness_sharing(population)

        adjusted = [gene.fitness.values[0] for gene in result]
        # ニッチ数4 → sqrt(4)=2 で優位幅を割る。最悪値は 1.0:
        # [1.0, 1.0+0.5, 1.0+1.0, 1.0+1.5]
        np.testing.assert_allclose(adjusted, [1.0, 1.5, 2.0, 2.5])

    def test_apply_fitness_sharing_never_improves_minimize_objective(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """最小化目的の混雑個体が共有調整で「改善」されないこと（方向バグ回帰）"""
        sharing = FitnessSharing(sharing_radius=0.1, alpha=1.0)
        monkeypatch.setattr(
            sharing, "_build_behavior_profile_map", lambda population: {}
        )
        monkeypatch.setattr(
            fitness_sharing,
            "_silhouette_based_sharing",
            lambda population, gene_serializer, vectorize_gene, objectives=None: (
                population
            ),
        )

        class FitnessAwareStrategyGene(StrategyGene):
            __slots__ = ("fitness",)

        population = []
        # 同一構造（ニッチ数4）のドローダウン（最小化）集団
        for i, drawdown in enumerate([0.40, 0.30, 0.20, 0.10]):
            gene = FitnessAwareStrategyGene(
                id=f"gene{i}",
                indicators=[IndicatorGene(type="SMA", parameters={"period": 20})],
                long_entry_conditions=[],
                short_entry_conditions=[],
                risk_management={},
                tpsl_gene=None,
                position_sizing_gene=None,
                metadata={},
            )
            gene.fitness = SimpleNamespace(values=(drawdown,), valid=True)
            population.append(gene)

        result = sharing.apply_fitness_sharing(population, objectives=["max_drawdown"])

        adjusted = [gene.fitness.values[0] for gene in result]
        # 旧実装はドローダウンを ニッチ数で割って改善させていた（0.1→0.05 等）。
        # 新実装では最悪個体（0.40）以外は必ず悪化（増加）する:
        # 最悪値 0.40 に向かって優位幅が 1/2 になる。
        np.testing.assert_allclose(adjusted, [0.40, 0.35, 0.30, 0.25])
        # 順序は維持される（最悪へ向かって縮むだけで逆転しない。
        # 入力が降順のため維持されるのも降順）
        assert adjusted == sorted(adjusted, key=lambda v: round(v, 9), reverse=True)

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


class TestTwoStageBehaviorThreshold:
    """二段階選抜の behavior 距離しきい値のテスト"""

    def test_threshold_uses_active_dimensions_only(self) -> None:
        from app.services.auto_strategy.core.engine.two_stage_selection import (
            TwoStageSelection,
        )

        selection = TwoStageSelection(
            toolbox=None,
            fitness_sharing=SimpleNamespace(sharing_radius=0.1),
        )

        # 実効1次元（残り9次元は全個体で0）
        vectors = {
            "a": np.zeros(10),
            "b": np.array([0.4] + [0.0] * 9),
            "c": np.array([1.0] + [0.0] * 9),
        }

        threshold = selection._get_behavior_distance_threshold(vectors)

        # 死んだ次元を含めた sqrt(10) ではなく sqrt(1) ベース
        assert threshold == pytest.approx(0.1)
        assert threshold < 0.4  # a-b 間の距離より小さい（別ニッチと判定される）

    def test_threshold_zero_when_all_dimensions_dead(self) -> None:
        from app.services.auto_strategy.core.engine.two_stage_selection import (
            TwoStageSelection,
        )

        selection = TwoStageSelection(
            toolbox=None,
            fitness_sharing=SimpleNamespace(sharing_radius=0.1),
        )

        vectors = {"a": np.zeros(5), "b": np.zeros(5)}

        assert selection._get_behavior_distance_threshold(vectors) == 0.0
