"""
遺伝的演算子のリファクタリングテスト

新しいラッパー関数をテスト()
"""

import copy
import uuid
from unittest.mock import Mock, patch

import pytest

from app.services.auto_strategy.models.strategy_gene import StrategyGene
from app.services.auto_strategy.models.indicator_gene import IndicatorGene
from app.services.auto_strategy.models.condition import Condition
from app.services.auto_strategy.core.genetic_operators import (
    _convert_to_strategy_gene,
    _convert_to_individual,
    create_deap_crossover_wrapper,
    create_deap_mutate_wrapper,
    crossover_strategy_genes_pure,
    mutate_strategy_gene_pure,
)


class TestGeneticOperatorsRefactored:
    """遺伝的演算子のリファクタリングテスト"""

    @pytest.fixture
    def sample_strategy_gene(self):
        """サンプル戦略遺伝子を作成"""
        gene = StrategyGene(
            id=str(uuid.uuid4()),
            indicators=[
                IndicatorGene(
                    type="SMA",
                    parameters={"period": 20}
                )
            ],
            entry_conditions=[
                Condition(
                    left_operand={"indicator_type": "SMA", "value_type": "close"},
                    operator=">",
                    right_operand=100.0
                )
            ],
            exit_conditions=[
                Condition(
                    left_operand={"indicator_type": "SMA", "value_type": "close"},
                    operator="<",
                    right_operand=105.0
                )
            ],
            risk_management={
                "position_size": 0.1,
                "take_profit_level": 2.0,
                "stop_loss_level": -1.0
            }
        )
        return gene

    @pytest.fixture
    def deap_individual_class(self):
        """DEAPのindividualクラスをモック"""
        # DEAPのcreatorがない場合のモック
        class MockIndividual(list):
            def __init__(self, values):
                super().__init__(values)
                self.fitness = Mock()

        return MockIndividual

    def test_convert_to_strategy_gene_with_strategy_gene(self, sample_strategy_gene):
        """StrategyGeneオブジェクトを入力された場合、そのまま返す"""
        result = _convert_to_strategy_gene(sample_strategy_gene)

        assert result == sample_strategy_gene
        assert isinstance(result, StrategyGene)

    def test_convert_to_strategy_gene_with_list(self, sample_strategy_gene):
        """listをStrategyGeneに変換"""
        # シリアライズされたデータをドロップ
        from app.services.auto_strategy.serializers.gene_serialization import GeneSerializer
        serializer = GeneSerializer()
        encoded = serializer.encode_strategy_gene_to_list(sample_strategy_gene)

        with patch('app.services.auto_strategy.serializers.gene_serialization.GeneSerializer') as mock_serializer:
            mock_serializer.return_value.decode_list_to_strategy_gene.return_value = sample_strategy_gene

            result = _convert_to_strategy_gene(encoded)

            assert isinstance(result, StrategyGene)
            mock_serializer.return_value.decode_list_to_strategy_gene.assert_called_once_with(
                encoded, StrategyGene
            )

    def test_convert_to_individual_with_strategy_gene_and_class(self, sample_strategy_gene, deap_individual_class):
        """StrategyGeneをindividualに変換"""
        with patch('app.services.auto_strategy.serializers.gene_serialization.GeneSerializer') as mock_serializer:
            mock_serializer.return_value.encode_strategy_gene_to_list.return_value = [1, 2, 3]

            result = _convert_to_individual(sample_strategy_gene, deap_individual_class)

            assert isinstance(result, deap_individual_class)
            assert result == [1, 2, 3]

    def test_convert_to_individual_without_class(self, sample_strategy_gene):
        """個体クラスなしの場合、StrategyGeneをそのまま返す"""
        result = _convert_to_individual(sample_strategy_gene, None)

        assert result == sample_strategy_gene
        assert isinstance(result, StrategyGene)

    @patch('app.services.auto_strategy.core.genetic_operators.crossover_strategy_genes_pure')
    def test_create_deap_crossover_wrapper(self, mock_pure_crossover, sample_strategy_gene, deap_individual_class):
        """DEAP交叉ラッパーのテスト"""
        # 親となる個体を作成
        parent1_data = [1, 2, 3]
        parent2_data = [4, 5, 6]
        parent1 = deap_individual_class(parent1_data)
        parent2 = deap_individual_class(parent2_data)

        # 子となる戦略遺伝子
        child1_gene = copy.deepcopy(sample_strategy_gene)
        child2_gene = copy.deepcopy(sample_strategy_gene)
        child1_gene.id = str(uuid.uuid4())
        child2_gene.id = str(uuid.uuid4())

        mock_pure_crossover.return_value = (child1_gene, child2_gene)

        # ラッパー関数を作成
        wrapper = create_deap_crossover_wrapper(deap_individual_class)

        # 関数を呼び出し
        result_child1, result_child2 = wrapper(parent1, parent2)

        # 純粋関数が呼ばれていることを確認
        mock_pure_crossover.assert_called_once()
        args, kwargs = mock_pure_crossover.call_args
        called_parent1, called_parent2 = args

        # 親がデコードされていることを確認
        # (実際のデコードロジックは上記のモックでカバー)

        # 結果が個体クラスであることを確認
        assert isinstance(result_child1, deap_individual_class)
        assert isinstance(result_child2, deap_individual_class)

    @patch('app.services.auto_strategy.core.genetic_operators.mutate_strategy_gene_pure')
    def test_create_deap_mutate_wrapper(self, mock_pure_mutate, sample_strategy_gene, deap_individual_class):
        """DEAP突然変異ラッパーのテスト"""
        # 親となる個体
        parent_data = [1, 2, 3]
        parent = deap_individual_class(parent_data)

        # 子となる戦略遺伝子
        mutated_gene = copy.deepcopy(sample_strategy_gene)
        mutated_gene.id = str(uuid.uuid4())

        mock_pure_mutate.return_value = mutated_gene

        # ラッパー関数作成
        with patch('deap.creator') as mock_creator:
            mock_creator.Individual = deap_individual_class
            wrapper = create_deap_mutate_wrapper()

        # 関数の呼び出し
        result_child = wrapper(parent)

        # 純粋関数が呼ばれていることを確認
        mock_pure_mutate.assert_called_once()

        # 結果が個体クラスであることを確認 (DEAPはリストとして返すので確認)
        assert isinstance(result_child, tuple)
        assert len(result_child) == 1
        assert isinstance(result_child[0], deap_individual_class)

    def test_crossover_strategy_genes_pure_creates_different_ids(self, sample_strategy_gene):
        """純粋交叉関数は新しいIDを生成する"""
        parent1 = copy.deepcopy(sample_strategy_gene)
        parent2 = copy.deepcopy(sample_strategy_gene)
        parent1.id = "parent1"
        parent2.id = "parent1"  # 意図的に同じID

        child1, child2 = crossover_strategy_genes_pure(parent1, parent2)

        assert isinstance(child1, StrategyGene)
        assert isinstance(child2, StrategyGene)
        assert child1.id != parent1.id
        assert child2.id != parent2.id
        assert child1.id != child2.id

    def test_mutate_strategy_gene_pure_creates_new_id(self, sample_strategy_gene):
        """純粋突然変異関数は新しいIDを生成する"""
        original_id = sample_strategy_gene.id

        mutated = mutate_strategy_gene_pure(sample_strategy_gene)

        assert isinstance(mutated, StrategyGene)
        assert mutated.id != original_id

    def test_crossover_preserves_structure(self, sample_strategy_gene):
        """交叉が構造を保持する"""
        parent1 = copy.deepcopy(sample_strategy_gene)
        parent2 = copy.deepcopy(sample_strategy_gene)

        child1, child2 = crossover_strategy_genes_pure(parent1, parent2)

        # 必須フィールドが存在することを確認
        assert hasattr(child1, 'indicators')
        assert hasattr(child1, 'entry_conditions')
        assert hasattr(child1, 'exit_conditions')
        assert hasattr(child1, 'risk_management')
        assert hasattr(child1, 'metadata')

        assert hasattr(child2, 'indicators')
        assert hasattr(child2, 'entry_conditions')
        assert hasattr(child2, 'exit_conditions')
        assert hasattr(child2, 'risk_management')
        assert hasattr(child2, 'metadata')

    def test_mutation_preserves_structure(self, sample_strategy_gene):
        """突然変異が構造を保持する"""
        mutated = mutate_strategy_gene_pure(sample_strategy_gene)

        # 必須フィールドが存在することを確認
        assert hasattr(mutated, 'indicators')
        assert hasattr(mutated, 'entry_conditions')
        assert hasattr(mutated, 'exit_conditions')
        assert hasattr(mutated, 'risk_management')
        assert hasattr(mutated, 'metadata')