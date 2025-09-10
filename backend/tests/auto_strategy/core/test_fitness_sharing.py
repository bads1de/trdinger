"""
テスト: fitness_sharing.py

FitnessSharingクラスのTDDテストケース
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from backend.app.services.auto_strategy.core.fitness_sharing import FitnessSharing
from backend.app.services.auto_strategy.models.strategy_models import StrategyGene, IndicatorGene


@pytest.fixture
def basic_fitness_sharing():
    """基本的なFitnessSharingインスタンス"""
    return FitnessSharing(sharing_radius=0.05, alpha=1.0)


@pytest.fixture
def mock_individual():
    """モック個体"""
    individual = MagicMock()
    individual.fitness.valid = True
    individual.fitness.values = (0.8,)
    return individual


@pytest.fixture
def test_strategy_gene_1():
    """テスト用StrategyGene"""
    indicator = IndicatorGene(type='SMA', parameters={'period': 50})
    gene = StrategyGene(
        indicators=[indicator],
        long_entry_conditions=[],
        short_entry_conditions=[],
        entry_conditions=[],
        exit_conditions=[],
        risk_management={'stop_loss': 0.01, 'take_profit': 0.02},
        tpsl_gene=None,
        position_sizing_gene=None
    )
    return gene


@pytest.fixture
def test_strategy_gene_2():
    """別のテスト用StrategyGene"""
    indicator = IndicatorGene(type='RSI', parameters={'period': 14})
    gene = StrategyGene(
        indicators=[indicator],
        long_entry_conditions=[],
        short_entry_conditions=[],
        entry_conditions=[],
        exit_conditions=[],
        risk_management={'stop_loss': 0.02, 'take_profit': 0.03},
        tpsl_gene=None,
        position_sizing_gene=None
    )
    return gene


class TestFitnessSharing:
    """FitnessSharingクラスのテスト"""

    def test_init(self, basic_fitness_sharing):
        """初期化テスト"""
        fs = basic_fitness_sharing
        assert fs.sharing_radius == 0.05
        assert fs.alpha == 1.0
        assert hasattr(fs, 'gene_serializer')
        assert fs.gene_serializer is not None

    def test_apply_fitness_sharing_single(self, basic_fitness_sharing, mock_individual):
        """単一個体の場合のテスト"""
        population = [mock_individual]
        result = basic_fitness_sharing.apply_fitness_sharing(population)
        assert result == population
        # 変更されていないことを確認
        assert mock_individual.fitness.values == (0.8,)

    def test_apply_fitness_sharing_invalid_fitness(self, basic_fitness_sharing, mock_individual):
        """無効なフィットネスを持つ個体のテスト"""
        mock_individual.fitness.valid = False
        population = [mock_individual]
        result = basic_fitness_sharing.apply_fitness_sharing(population)
        assert result == population

    @patch('backend.app.services.auto_strategy.serializers.gene_serialization.GeneSerializer')
    def test_apply_fitness_sharing_with_mock(self, mock_gene_serializer_class, basic_fitness_sharing, mock_individual):
        """GeneSerializerをモックしたテスト"""
        mock_serializer = Mock()
        mock_gene_serializer_class.return_value = mock_serializer

        # デコード失敗の場合
        mock_serializer.from_list.side_effect = Exception("Decode error")

        population = [mock_individual]
        result = basic_fitness_sharing.apply_fitness_sharing(population)
        assert result == population
        # from_list が呼び出されたことを確認
        mock_serializer.from_list.assert_called()

    def test_calculate_similarity_basic(self, basic_fitness_sharing, test_strategy_gene_1, test_strategy_gene_2):
        """基本的な類似度計算テスト"""
        similarity = basic_fitness_sharing._calculate_similarity(test_strategy_gene_1, test_strategy_gene_2)
        # 結果は0.0-1.0の範囲
        assert 0.0 <= similarity <= 1.0

    def test_sharing_function_inside_radius(self, basic_fitness_sharing):
        """共有関数テスト - 半径内"""
        # sharing_radiusは0.05なので0.03は内
        result = basic_fitness_sharing._sharing_function(0.03)
        assert result == 1.0

    def test_sharing_function_outside_radius(self, basic_fitness_sharing):
        """共有関数テスト - 半径外"""
        # 0.1は半径外
        result = basic_fitness_sharing._sharing_function(0.1)
        assert result == 0.0

    def test_sharing_function_negative_similarity(self, basic_fitness_sharing):
        """共有関数テスト - 負の類似度（バグ修正確認）"""
        # 修正後: 負のsimilarityは0.0を返す
        result = basic_fitness_sharing._sharing_function(-0.1)
        assert result == 0.0

    def test_calculate_indicator_similarity_same(self, basic_fitness_sharing, test_strategy_gene_1):
        """同一指標の類似度テスト"""
        similarity = basic_fitness_sharing._calculate_indicator_similarity(test_strategy_gene_1.indicators, test_strategy_gene_1.indicators)
        assert similarity == 1.0

    def test_calculate_indicator_similarity_different(self, basic_fitness_sharing, test_strategy_gene_1, test_strategy_gene_2):
        """異なる指標の類似度テスト"""
        similarity = basic_fitness_sharing._calculate_indicator_similarity(test_strategy_gene_1.indicators, test_strategy_gene_2.indicators)
        assert similarity == 0.0  # 異なるタイプ

    def test_calculate_condition_similarity_empty(self, basic_fitness_sharing):
        """空条件の類似度テスト"""
        similarity = basic_fitness_sharing._calculate_condition_similarity([], [])
        assert similarity == 1.0

    def test_calculate_risk_management_similarity(self, basic_fitness_sharing, test_strategy_gene_1, test_strategy_gene_2):
        """リスク管理の類似度テスト"""
        similarity = basic_fitness_sharing._calculate_risk_management_similarity(
            test_strategy_gene_1.risk_management,
            test_strategy_gene_2.risk_management
        )
        assert 0.0 <= similarity <= 1.0

    def test_calculate_tpsl_similarity_none(self, basic_fitness_sharing):
        """TP/SLなしの類似度テスト"""
        similarity = basic_fitness_sharing._calculate_tpsl_similarity(None, None)
        assert similarity == 1.0

    def test_calculate_position_sizing_similarity_none(self, basic_fitness_sharing):
        """ポジションサイジングなしの類似度テスト"""
        similarity = basic_fitness_sharing._calculate_position_sizing_similarity(None, None)


class TestFitnessSharingAdditional:
    """追加のFitnessSharingテスト"""

    def test_large_population_fitness_sharing(self, basic_fitness_sharing):
        """大規模個体群でのフィットネス共有テスト"""
        # 100個の個体を作成
        population = []
        for i in range(100):
            individual = basic_fitness_sharing.gene_serializer.to_list(test_strategy_gene_1)
            individual.fitness.valid = True
            individual.fitness.values = (0.8 + i * 0.01, 0.5)
            population.append(individual)

        result = basic_fitness_sharing.apply_fitness_sharing(population)

        # 結果が人口と同じ長さであること確認
        assert len(result) == 100

        # 各個体のフィットネスが適切に調整されていることを確認
        for ind in result:
            if hasattr(ind, "fitness") and ind.fitness.valid:
                for fitness_val in ind.fitness.values:
                    assert isinstance(fitness_val, (int, float))

    def test_division_by_zero_protection(self):
        """ゼロ除算保護テスト"""
        # niche_countが最小1.0に保護されていることを確認
        fs = FitnessSharing(sharing_radius=0.01, alpha=1.0)

        individual = fs.gene_serializer.to_list(test_strategy_gene_1)
        individual.fitness.valid = True
        individual.fitness.values = (1.0,)

        population = [individual]

        result = fs.apply_fitness_sharing(population)

        # 処理が完了し、結果が返されること確認
        assert result is not None
        assert len(result) == 1

        # フィットネス値が正しく調整されること
        if hasattr(result[0], "fitness") and result[0].fitness.valid:
            for fv in result[0].fitness.values:
                assert isinstance(fv, (int, float))


class TestFitnessSharingBugFixes:
    """バグ修正確認テスト"""

    def test_similarity_range_clamping(self, basic_fitness_sharing):
        """類似度範囲クリッピングテスト"""
        # 範囲外の値を返すようにパッチを適用
        with patch.object(basic_fitness_sharing, '_calculate_indicator_similarity', return_value=-0.5):
            similarity = basic_fitness_sharing._calculate_similarity(test_strategy_gene_1, test_strategy_gene_2)
            assert similarity == 0.0  # 最低値0.0にクリップされているはず

        with patch.object(basic_fitness_sharing, '_calculate_indicator_similarity', return_value=1.5):
            similarity = basic_fitness_sharing._calculate_similarity(test_strategy_gene_1, test_strategy_gene_2)
            assert similarity == 1.0  # 最大値1.0にクリップされているはず


if __name__ == "__main__":
    pytest.main([__file__])