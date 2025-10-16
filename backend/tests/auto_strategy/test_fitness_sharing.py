"""
フィットネス共有のテスト
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from app.services.auto_strategy.core.fitness_sharing import FitnessSharing
from app.services.auto_strategy.models.strategy_models import (
    StrategyGene,
    IndicatorGene,
    Condition,
    TPSLGene,
    PositionSizingGene,
)


class TestFitnessSharing:
    """フィットネス共有のテスト"""

    @pytest.fixture
    def fitness_sharing(self):
        """FitnessSharingインスタンス"""
        return FitnessSharing(sharing_radius=0.1, alpha=1.0)

    @pytest.fixture
    def sample_population(self):
        """サンプル個体群"""

        # DEAP形式のモック個体を作成
        class MockIndividual(list):
            def __init__(self, gene, fitness_values):
                super().__init__([gene])
                self.fitness = Mock()
                self.fitness.values = fitness_values
                self.fitness.valid = True

        return [
            MockIndividual(
                StrategyGene(
                    id="gene1",
                    indicators=[IndicatorGene(type="SMA", parameters={"period": 10})],
                    entry_conditions=[
                        Condition(
                            left_operand="close", operator=">", right_operand="sma"
                        )
                    ],
                    exit_conditions=[],
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
            MockIndividual(
                StrategyGene(
                    id="gene2",
                    indicators=[IndicatorGene(type="EMA", parameters={"period": 20})],
                    entry_conditions=[
                        Condition(
                            left_operand="close", operator="<", right_operand="ema"
                        )
                    ],
                    exit_conditions=[],
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
            MockIndividual(
                StrategyGene(
                    id="gene3",
                    indicators=[IndicatorGene(type="RSI", parameters={"period": 14})],
                    entry_conditions=[
                        Condition(left_operand="rsi", operator="<", right_operand="30")
                    ],
                    exit_conditions=[],
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

    def test_apply_fitness_sharing_basic(self, fitness_sharing, sample_population):
        """基本的なフィットネス共有適用テスト"""
        original_fitness = [ind.fitness.values for ind in sample_population]

        result = fitness_sharing.apply_fitness_sharing(sample_population)

        assert len(result) == len(sample_population)
        # フィットネスが調整されているはず
        adjusted_fitness = [ind.fitness.values for ind in result]
        assert adjusted_fitness != original_fitness  # 少なくとも何かが変わっている

    def test_silhouette_based_sharing_basic(self, fitness_sharing, sample_population):
        """シルエットベース共有の基本テスト"""
        # 関数が存在するか確認（実装前にテストを書く）
        if hasattr(fitness_sharing, "silhouette_based_sharing"):
            original_fitness = [ind.fitness.values for ind in sample_population]

            result = fitness_sharing.silhouette_based_sharing(sample_population)

            assert len(result) == len(sample_population)
            # シルエットスコアに基づいて調整されているはず
            adjusted_fitness = [ind.fitness.values for ind in result]
            # 具体的な検証は実装後に追加
            pass
        else:
            pytest.skip("silhouette_based_sharing method not implemented yet")

    def test_silhouette_based_sharing_clustering(
        self, fitness_sharing, sample_population
    ):
        """シルエットベース共有のクラスタリング検証"""
        if hasattr(fitness_sharing, "silhouette_based_sharing"):
            with (
                patch("sklearn.cluster.KMeans") as mock_kmeans,
                patch("sklearn.metrics.silhouette_samples") as mock_silhouette,
            ):

                # KMeansのモック
                mock_kmeans_instance = Mock()
                mock_kmeans.return_value = mock_kmeans_instance
                mock_kmeans_instance.fit_predict.return_value = [0, 0, 1]  # 2クラスタ

                # silhouette_samplesのモック
                mock_silhouette.return_value = [0.5, 0.6, -0.1]  # サンプルスコア

                result = fitness_sharing.silhouette_based_sharing(sample_population)

                # KMeansが呼ばれたか確認
                mock_kmeans.assert_called_once()
                mock_silhouette.assert_called_once()

                # フィットネスが調整されているか確認
                for ind in result:
                    assert hasattr(ind.fitness, "values")
        else:
            pytest.skip("silhouette_based_sharing method not implemented yet")

    def test_silhouette_based_sharing_fitness_adjustment(
        self, fitness_sharing, sample_population
    ):
        """シルエットスコアに基づくフィットネス調整テスト"""
        if hasattr(fitness_sharing, "silhouette_based_sharing"):
            # シルエットスコアが低い個体のfitnessがより強く調整されるはず
            pass  # 実装後に具体的にテスト
        else:
            pytest.skip("silhouette_based_sharing method not implemented yet")

    def test_apply_fitness_sharing_with_silhouette(
        self, fitness_sharing, sample_population
    ):
        """拡張されたapply_fitness_sharingのテスト（シルエット適用後）"""
        if hasattr(fitness_sharing, "silhouette_based_sharing"):
            original_fitness = [ind.fitness.values for ind in sample_population]

            result = fitness_sharing.apply_fitness_sharing(sample_population)

            assert len(result) == len(sample_population)
            # シルエットベース調整が適用されているはず
            # 具体的な検証は実装後に
            pass
        else:
            pytest.skip("silhouette_based_sharing method not implemented yet")
