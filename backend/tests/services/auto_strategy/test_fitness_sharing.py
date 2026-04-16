import copy
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.services.auto_strategy.core.fitness.fitness_sharing import FitnessSharing
from app.services.auto_strategy.genes import (
    Condition,
    ConditionGroup,
    IndicatorGene,
    PositionSizingGene,
    StrategyGene,
    TPSLGene,
)


class TestFitnessSharing:
    """フィットネス共有の包括的なテスト (基本動作、最適化、サンプリング)"""

    @pytest.fixture
    def fitness_sharing(self):
        """FitnessSharingインスタンス"""
        return FitnessSharing(sharing_radius=0.1, alpha=1.0)

    @pytest.fixture
    def sample_population(self, sample_strategy_gene):
        """サンプル個体群"""

        return [
            self._create_mock_individual(
                StrategyGene(
                    id="gene1",
                    indicators=[IndicatorGene(type="SMA", parameters={"period": 10})],
                    long_entry_conditions=[
                        Condition(
                            left_operand="close", operator=">", right_operand="sma"
                        )
                    ],
                    short_entry_conditions=[],
                    risk_management={"position_size": 0.1},
                    tpsl_gene=TPSLGene(),
                    position_sizing_gene=PositionSizingGene(),
                    metadata={},
                ),
                (1.0, 0.5),
            ),
            self._create_mock_individual(
                StrategyGene(
                    id="gene2",
                    indicators=[IndicatorGene(type="EMA", parameters={"period": 20})],
                    long_entry_conditions=[
                        Condition(
                            left_operand="close", operator="<", right_operand="ema"
                        )
                    ],
                    short_entry_conditions=[],
                    risk_management={"position_size": 0.2},
                    tpsl_gene=TPSLGene(),
                    position_sizing_gene=PositionSizingGene(),
                    metadata={},
                ),
                (0.8, 0.6),
            ),
            self._create_mock_individual(
                StrategyGene(
                    id="gene3",
                    indicators=[IndicatorGene(type="RSI", parameters={"period": 14})],
                    long_entry_conditions=[
                        Condition(left_operand="rsi", operator="<", right_operand="30")
                    ],
                    short_entry_conditions=[],
                    risk_management={"position_size": 0.15},
                    tpsl_gene=TPSLGene(),
                    position_sizing_gene=PositionSizingGene(),
                    metadata={},
                ),
                (0.9, 0.7),
            ),
        ]

    @pytest.fixture
    def sample_strategy_gene(self):
        """サンプル戦略遺伝子"""
        return StrategyGene(
            id="test_gene",
            indicators=[IndicatorGene(type="SMA", parameters={"period": 10})],
            risk_management={"position_size": 0.1},
            tpsl_gene=TPSLGene(),
            position_sizing_gene=PositionSizingGene(),
            metadata={},
        )

    def _create_mock_individual(self, gene: StrategyGene, fitness_values: tuple):
        """モック個体を作成"""

        class MockIndividual(list):
            def __init__(self, gene, fitness_values):
                super().__init__([gene])
                self.fitness = Mock()
                self.fitness.values = fitness_values
                self.fitness.valid = True

        return MockIndividual(gene, fitness_values)

    def _create_large_population(self, base_gene: StrategyGene, size: int) -> list:
        """大規模な個体群を生成"""
        population = []
        for i in range(size):
            gene = copy.deepcopy(base_gene)
            gene.id = f"gene_{i}"
            gene.risk_management = {"position_size": 0.05 + (i % 10) * 0.02}
            if gene.tpsl_gene:
                gene.tpsl_gene.stop_loss_pct = 0.01 + (i % 5) * 0.01
                gene.tpsl_gene.take_profit_pct = 0.02 + (i % 8) * 0.01

            fitness = (0.5 + np.random.random() * 0.5,)
            population.append(self._create_mock_individual(gene, fitness))
        return population

    # ---------------------------------------------------------------------------
    # 基本動作テスト
    # ---------------------------------------------------------------------------

    def test_apply_fitness_sharing_basic(self, fitness_sharing, sample_population):
        """基本属性と実行のテスト"""
        original_fitness = [ind.fitness.values for ind in sample_population]
        result = fitness_sharing.apply_fitness_sharing(sample_population)
        assert len(result) == len(sample_population)
        assert len([ind.fitness.values for ind in result]) == len(original_fitness)

    def test_apply_fitness_sharing_empty_population(self, fitness_sharing):
        """空の個体群に対する処理"""
        assert fitness_sharing.apply_fitness_sharing([]) == []

    def test_apply_fitness_sharing_single_individual(
        self, fitness_sharing, sample_strategy_gene
    ):
        """単一個体に対する処理"""
        population = self._create_large_population(sample_strategy_gene, 1)
        result = fitness_sharing.apply_fitness_sharing(population)
        assert len(result) == 1

    # ---------------------------------------------------------------------------
    # ベクトル化テスト
    # ---------------------------------------------------------------------------

    def test_vectorize_gene_enhanced(self, fitness_sharing):
        """拡張ベクトル化のテスト (指標、オペレータ、期間など)"""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                "app.services.auto_strategy.core.fitness.fitness_sharing.get_valid_indicator_types",
                lambda: ["SMA", "EMA", "RSI", "MACD"],
            )
            fitness_sharing.__init__(sharing_radius=0.1)

            gene1 = StrategyGene(
                id="gene1",
                indicators=[
                    IndicatorGene(type="SMA", parameters={"period": 10}),
                    IndicatorGene(type="RSI", parameters={"period": 14}),
                ],
                long_entry_conditions=[],
                short_entry_conditions=[],
                risk_management={},
                tpsl_gene=None,
                position_sizing_gene=None,
                metadata={},
            )
            vec1 = fitness_sharing._vectorize_gene(gene1)
            assert len(vec1) >= 15

            # 指標部分の確認 (SMA=1, RSI=1)
            indicator_start_idx = 7
            assert vec1[indicator_start_idx + 2] == 1.0  # RSI
            assert vec1[indicator_start_idx + 3] == 1.0  # SMA

    def test_vectorize_condition_group(self, fitness_sharing):
        """ConditionGroupのベクトル化テスト"""
        fitness_sharing.__init__(sharing_radius=0.1)
        gene = StrategyGene(
            id="gene_group",
            indicators=[],
            long_entry_conditions=[
                ConditionGroup(
                    operator="AND",
                    conditions=[
                        Condition(
                            left_operand="close", operator=">", right_operand="sma"
                        ),
                        ConditionGroup(
                            operator="OR",
                            conditions=[
                                Condition(
                                    left_operand="rsi", operator="<", right_operand="30"
                                ),
                                Condition(
                                    left_operand="adx", operator=">", right_operand="25"
                                ),
                            ],
                        ),
                    ],
                )
            ],
            short_entry_conditions=[],
            risk_management={},
            tpsl_gene=None,
            position_sizing_gene=None,
            metadata={},
        )
        vec = fitness_sharing._vectorize_gene(gene)

        op_start_idx = 7 + (
            len(fitness_sharing.indicator_types)
            if fitness_sharing.indicator_types
            else 0
        )
        and_idx = fitness_sharing.operator_map["AND"]
        or_idx = fitness_sharing.operator_map["OR"]
        assert vec[op_start_idx + and_idx] >= 1.0
        assert vec[op_start_idx + or_idx] >= 1.0

    # ---------------------------------------------------------------------------
    # 最適化 & ベクトル化計算テスト
    # ---------------------------------------------------------------------------

    def test_compute_niche_counts_vectorized_consistency(
        self, fitness_sharing, sample_strategy_gene
    ):
        """ベクトル化ニッチカウント計算の整合性確認"""
        population = self._create_large_population(sample_strategy_gene, 10)
        vectors = np.array(
            [fitness_sharing._vectorize_gene(ind[0]) for ind in population]
        )

        niche_counts = fitness_sharing.compute_niche_counts_vectorized(vectors)
        assert len(niche_counts) == len(vectors)
        assert all(nc >= 1.0 for nc in niche_counts)

    def test_kdtree_neighbor_search(self, fitness_sharing):
        """KD-Tree による近傍探索テスト"""
        vectors = np.random.rand(20, 7)
        neighbors_list = fitness_sharing.find_neighbors_kdtree(vectors, radius=0.5)
        assert len(neighbors_list) == len(vectors)
        for neighbors in neighbors_list:
            assert len(neighbors) >= 1

    # ---------------------------------------------------------------------------
    # パフォーマンス & サンプリングテスト
    # ---------------------------------------------------------------------------

    def test_large_population_sampling(self, fitness_sharing, sample_strategy_gene):
        """サンプリング閾値設定と大規模個体群での動作"""
        fitness_sharing.sampling_threshold = 10
        population = self._create_large_population(sample_strategy_gene, 20)
        vectors = np.array(
            [fitness_sharing._vectorize_gene(ind[0]) for ind in population]
        )

        niche_counts = fitness_sharing.compute_niche_counts_vectorized(vectors)
        assert len(niche_counts) == 20
        assert all(nc >= 1.0 for nc in niche_counts)

    def test_apply_fitness_sharing_performance(
        self, fitness_sharing, sample_strategy_gene
    ):
        """パフォーマンス計測 (100個体)"""
        population_size = 100
        population = self._create_large_population(
            sample_strategy_gene, population_size
        )

        start_time = time.time()
        result = fitness_sharing.apply_fitness_sharing(population)
        elapsed_time = time.time() - start_time

        assert len(result) == population_size
        assert elapsed_time < 5.0  # 5秒以内を期待

    def test_apply_fitness_sharing_reuses_cached_vectors_for_silhouette_phase(
        self,
        fitness_sharing,
        sample_population,
        monkeypatch,
    ):
        """シルエット段でも同じベクトルキャッシュを再利用すること."""
        vectorize_calls = {"count": 0}
        original_vectorize = fitness_sharing._vectorize_gene

        def counted_vectorize(gene, behavior_profile=None):
            vectorize_calls["count"] += 1
            return original_vectorize(gene, behavior_profile=behavior_profile)

        def fake_silhouette(population, gene_serializer, vectorize_gene):
            for individual in population:
                gene = gene_serializer.from_list(individual, StrategyGene)
                vectorize_gene(gene)
            return population

        monkeypatch.setattr(fitness_sharing, "_vectorize_gene", counted_vectorize)
        monkeypatch.setattr(
            fitness_sharing,
            "compute_niche_counts_vectorized",
            lambda vectors: np.ones(len(vectors)),
        )
        monkeypatch.setattr(
            "app.services.auto_strategy.core.fitness.fitness_sharing._silhouette_based_sharing",
            fake_silhouette,
        )

        result = fitness_sharing.apply_fitness_sharing(sample_population)

        assert len(result) == len(sample_population)
        assert vectorize_calls["count"] == len(sample_population)

    def test_apply_fitness_sharing_reflects_behavior_report_even_for_same_genotype(
        self,
        monkeypatch,
    ):
        """同じ genotype でも report が違えば別ベクトルとして扱うこと."""
        gene1 = StrategyGene(
            id="gene_same_1",
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
        gene2 = copy.deepcopy(gene1)
        gene2.id = "gene_same_2"

        individual1 = self._create_mock_individual(gene1, (1.0,))
        individual2 = self._create_mock_individual(gene2, (1.0,))

        sharing = FitnessSharing(sharing_radius=0.1, alpha=1.0)
        sharing.set_evaluation_report_provider(
            lambda individual: {
                "gene_same_1": SimpleNamespace(
                    pass_rate=1.0,
                    primary_aggregated_fitness=1.0,
                    primary_worst_case_fitness=0.9,
                    scenarios=[
                        SimpleNamespace(
                            performance_metrics={
                                "total_return": 0.15,
                                "sharpe_ratio": 1.8,
                                "max_drawdown": 0.05,
                                "total_trades": 24,
                            }
                        )
                    ],
                ),
                "gene_same_2": SimpleNamespace(
                    pass_rate=0.0,
                    primary_aggregated_fitness=1.0,
                    primary_worst_case_fitness=-0.4,
                    scenarios=[
                        SimpleNamespace(
                            performance_metrics={
                                "total_return": -0.05,
                                "sharpe_ratio": -0.2,
                                "max_drawdown": 0.32,
                                "total_trades": 4,
                            }
                        )
                    ],
                ),
            }[individual[0].id]
        )

        captured = {}

        def capture_vectors(vectors):
            captured["vectors"] = vectors.copy()
            return np.ones(len(vectors))

        monkeypatch.setattr(sharing, "compute_niche_counts_vectorized", capture_vectors)
        monkeypatch.setattr(
            "app.services.auto_strategy.core.fitness.fitness_sharing._silhouette_based_sharing",
            lambda population, gene_serializer, vectorize_gene: population,
        )

        sharing.apply_fitness_sharing([individual1, individual2])

        assert "vectors" in captured
        assert captured["vectors"].shape[0] == 2
        assert not np.array_equal(captured["vectors"][0], captured["vectors"][1])

    def test_feature_vector_cache_key_changes_after_in_place_gene_mutation(
        self,
        fitness_sharing,
        sample_strategy_gene,
    ):
        """同一個体の内容変更時は特徴ベクトルキャッシュキーも変わること."""
        first_key = fitness_sharing._get_feature_vector_cache_key(sample_strategy_gene)

        sample_strategy_gene.risk_management["position_size"] = 0.25

        second_key = fitness_sharing._get_feature_vector_cache_key(sample_strategy_gene)

        assert first_key != second_key
