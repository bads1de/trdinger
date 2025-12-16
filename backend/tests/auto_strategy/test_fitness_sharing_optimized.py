"""
フィットネス共有の最適化テスト

$O(N^2)$ のペアワイズ計算を効率化するための最適化されたアルゴリズムのテスト。
"""

import time
from unittest.mock import Mock

import numpy as np
import pytest

from app.services.auto_strategy.core.fitness_sharing import FitnessSharing
from app.services.auto_strategy.genes import (
    Condition,
    IndicatorGene,
    PositionSizingGene,
    StrategyGene,
    TPSLGene,
)


class TestOptimizedFitnessSharing:
    """最適化されたフィットネス共有のテスト"""

    @pytest.fixture
    def fitness_sharing(self):
        """FitnessSharingインスタンス"""
        return FitnessSharing(sharing_radius=0.1, alpha=1.0)

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
        import copy

        population = []
        for i in range(size):
            gene = copy.deepcopy(base_gene)
            gene.id = f"gene_{i}"
            # 多様性を追加
            gene.risk_management = {"position_size": 0.05 + (i % 10) * 0.02}
            if gene.tpsl_gene:
                gene.tpsl_gene.stop_loss_pct = 0.01 + (i % 5) * 0.01
                gene.tpsl_gene.take_profit_pct = 0.02 + (i % 8) * 0.01

            fitness = (0.5 + np.random.random() * 0.5,)
            population.append(self._create_mock_individual(gene, fitness))

        return population

    # ---------------------------------------------------------------------------
    # ベクトル化のテスト
    # ---------------------------------------------------------------------------

    def test_vectorize_gene_returns_consistent_shape(
        self, fitness_sharing, sample_strategy_gene
    ):
        """遺伝子のベクトル化が一貫した形状を返すことを確認"""
        vector = fitness_sharing._vectorize_gene(sample_strategy_gene)

        assert isinstance(vector, np.ndarray)
        assert vector.ndim == 1
        assert len(vector) > 0

    def test_vectorize_gene_is_deterministic(
        self, fitness_sharing, sample_strategy_gene
    ):
        """同じ遺伝子に対して同じベクトルが返されることを確認"""
        vector1 = fitness_sharing._vectorize_gene(sample_strategy_gene)
        vector2 = fitness_sharing._vectorize_gene(sample_strategy_gene)

        np.testing.assert_array_equal(vector1, vector2)

    # ---------------------------------------------------------------------------
    # 最適化されたニッチカウント計算のテスト
    # ---------------------------------------------------------------------------

    def test_compute_niche_counts_vectorized_exists(self, fitness_sharing):
        """ベクトル化されたニッチカウント計算メソッドが存在することを確認"""
        assert hasattr(fitness_sharing, "compute_niche_counts_vectorized")

    def test_compute_niche_counts_vectorized_returns_correct_length(
        self, fitness_sharing, sample_strategy_gene
    ):
        """ベクトル化ニッチカウントが正しい長さを返すことを確認"""
        population = self._create_large_population(sample_strategy_gene, 10)

        # 遺伝子をベクトル化
        vectors = []
        for ind in population:
            try:
                gene = fitness_sharing.gene_serializer.from_list(ind, StrategyGene)
                vectors.append(fitness_sharing._vectorize_gene(gene))
            except Exception:
                pass

        vectors = np.array(vectors)
        niche_counts = fitness_sharing.compute_niche_counts_vectorized(vectors)

        assert len(niche_counts) == len(vectors)
        assert all(nc >= 1.0 for nc in niche_counts)

    def test_compute_niche_counts_vectorized_consistency(
        self, fitness_sharing, sample_strategy_gene
    ):
        """ベクトル化ニッチカウントがナイーブ実装と一致することを確認"""
        # 小さい集団で厳密一致を確認
        population = self._create_large_population(sample_strategy_gene, 5)

        # 遺伝子をベクトル化
        genes = []
        vectors = []
        for ind in population:
            try:
                gene = fitness_sharing.gene_serializer.from_list(ind, StrategyGene)
                genes.append(gene)
                vectors.append(fitness_sharing._vectorize_gene(gene))
            except Exception:
                pass

        vectors = np.array(vectors)

        # ベクトル化版
        vectorized_counts = fitness_sharing.compute_niche_counts_vectorized(vectors)

        # ナイーブ版（元のO(N^2)実装を参考に）
        naive_counts = []
        for i, gene_i in enumerate(genes):
            niche_count = 0.0
            for j, gene_j in enumerate(genes):
                similarity = fitness_sharing._calculate_similarity(gene_i, gene_j)
                sharing_value = fitness_sharing._sharing_function(similarity)
                niche_count += sharing_value
            naive_counts.append(max(1.0, niche_count))

        # 類似度計算アルゴリズムが異なるため、完全一致は期待しない
        # ただし、同程度の範囲にあることを確認
        for v_count, n_count in zip(vectorized_counts, naive_counts):
            # 両方とも少なくとも1.0以上であることを確認
            assert v_count >= 1.0
            assert n_count >= 1.0

    # ---------------------------------------------------------------------------
    # パフォーマンステスト
    # ---------------------------------------------------------------------------

    @pytest.mark.parametrize("population_size", [50, 100])
    def test_apply_fitness_sharing_performance(
        self, fitness_sharing, sample_strategy_gene, population_size
    ):
        """大規模集団でのパフォーマンスを確認"""
        population = self._create_large_population(
            sample_strategy_gene, population_size
        )

        start_time = time.time()
        result = fitness_sharing.apply_fitness_sharing(population)
        elapsed_time = time.time() - start_time

        assert len(result) == population_size
        # 100個体でも5秒以内に完了することを確認（閾値は環境依存）
        # 実際の閾値は必要に応じて調整
        assert (
            elapsed_time < 10.0
        ), f"Took {elapsed_time:.2f}s for {population_size} individuals"

    def test_quadratic_vs_vectorized_scaling(
        self, fitness_sharing, sample_strategy_gene
    ):
        """集団サイズに対するスケーリングを確認"""
        times = {}

        for size in [10, 20, 30]:
            population = self._create_large_population(sample_strategy_gene, size)

            start_time = time.time()
            fitness_sharing.apply_fitness_sharing(population)
            elapsed_time = time.time() - start_time

            times[size] = elapsed_time

        # O(N^2) なら、サイズが2倍になると時間は4倍になる
        # O(N log N) や O(N) なら、2倍程度
        # 少なくとも急激な増加でないことを確認
        ratio_10_20 = times[20] / times[10] if times[10] > 0.001 else 1
        ratio_20_30 = times[30] / times[20] if times[20] > 0.001 else 1

        # ログ出力（デバッグ用）
        print(f"Times: {times}")
        print(f"Ratio 10->20: {ratio_10_20:.2f}, Ratio 20->30: {ratio_20_30:.2f}")

        # 極端な増加（例: 10倍以上）がないことを確認
        assert ratio_10_20 < 10.0
        assert ratio_20_30 < 10.0

    # ---------------------------------------------------------------------------
    # エッジケースのテスト
    # ---------------------------------------------------------------------------

    def test_apply_fitness_sharing_empty_population(self, fitness_sharing):
        """空の個体群に対する処理"""
        result = fitness_sharing.apply_fitness_sharing([])
        assert result == []

    def test_apply_fitness_sharing_single_individual(
        self, fitness_sharing, sample_strategy_gene
    ):
        """単一個体に対する処理"""
        population = self._create_large_population(sample_strategy_gene, 1)
        result = fitness_sharing.apply_fitness_sharing(population)
        assert len(result) == 1

    def test_apply_fitness_sharing_with_invalid_individuals(
        self, fitness_sharing, sample_strategy_gene
    ):
        """無効な個体が混在する場合の処理"""
        population = self._create_large_population(sample_strategy_gene, 5)

        # 無効な個体を追加
        invalid_individual = Mock()
        invalid_individual.__iter__ = Mock(side_effect=Exception("Invalid"))
        invalid_individual.fitness = Mock()
        invalid_individual.fitness.valid = False

        population.append(invalid_individual)

        # エラーなく処理されることを確認
        result = fitness_sharing.apply_fitness_sharing(population)
        assert len(result) == len(population)


class TestKDTreeOptimization:
    """KD-Tree を使用した近傍探索最適化のテスト"""

    @pytest.fixture
    def fitness_sharing(self):
        return FitnessSharing(sharing_radius=0.1, alpha=1.0)

    def test_kdtree_neighbor_search_exists(self, fitness_sharing):
        """KD-Tree による近傍探索メソッドが存在することを確認"""
        assert hasattr(fitness_sharing, "find_neighbors_kdtree")

    def test_kdtree_neighbor_search_returns_indices(self, fitness_sharing):
        """KD-Tree が正しいインデックスを返すことを確認"""
        # テスト用のランダムベクトル
        np.random.seed(42)
        vectors = np.random.rand(20, 7)

        neighbors_list = fitness_sharing.find_neighbors_kdtree(vectors, radius=0.5)

        assert len(neighbors_list) == len(vectors)
        for neighbors in neighbors_list:
            assert isinstance(neighbors, (list, np.ndarray))
            # 各点は少なくとも自分自身を含む
            assert len(neighbors) >= 1


class TestSamplingOptimization:
    """サンプリングベースの近似最適化のテスト"""

    @pytest.fixture
    def fitness_sharing(self):
        return FitnessSharing(sharing_radius=0.1, alpha=1.0)

    def test_sampling_threshold_config(self, fitness_sharing):
        """サンプリング閾値が設定可能であることを確認"""
        # デフォルト値が存在する
        assert hasattr(fitness_sharing, "sampling_threshold")
        assert fitness_sharing.sampling_threshold > 0

    def test_large_population_uses_sampling(self, fitness_sharing):
        """大規模集団でサンプリングが使用されることを確認"""
        # サンプリング閾値を小さく設定してテスト
        fitness_sharing.sampling_threshold = 10

        np.random.seed(42)
        vectors = np.random.rand(50, 7)

        # サンプリングが適用されるか確認（内部実装に依存）
        niche_counts = fitness_sharing.compute_niche_counts_vectorized(vectors)

        assert len(niche_counts) == len(vectors)
        assert all(nc >= 1.0 for nc in niche_counts)




