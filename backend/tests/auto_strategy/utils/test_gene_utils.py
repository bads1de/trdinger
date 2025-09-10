"""gene_utilsクラスのテストモジュール"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import uuid
from app.services.auto_strategy.utils.gene_utils import BaseGene, GeneticUtils, GeneUtils

class TestBaseGene:
    """BaseGeneクラスのテスト"""

    def test_to_dict_basic_attributes(self):
        """基本属性の辞書変換テスト"""
        class TestGene(BaseGene):
            def __init__(self):
                self.public_attr = "test"
                self._private_attr = "private"
                self.number_attr = 42

            def _validate_parameters(self, errors):
                pass

        gene = TestGene()
        result = gene.to_dict()

        assert "public_attr" in result
        assert "_private_attr" not in result
        assert result["public_attr"] == "test"
        assert result["number_attr"] == 42

    def test_to_dict_with_enum(self):
        """Enum属性の辞書変換テスト"""
        from enum import Enum

        class TestEnum(Enum):
            VALUE1 = "value1"
            VALUE2 = "value2"

        class TestGene(BaseGene):
            def __init__(self):
                self.enum_attr = TestEnum.VALUE1
                self.enabled = True

            def _validate_parameters(self, errors):
                pass

        gene = TestGene()
        result = gene.to_dict()

        assert result["enum_attr"] == "value1"
        assert result["enabled"] is True

    def test_to_dict_with_datetime(self):
        """datetime属性の辞書変換テスト"""
        class TestGene(BaseGene):
            def __init__(self):
                self.timestamp = datetime(2023, 1, 1, 12, 0, 0)

            def _validate_parameters(self, errors):
                pass

        gene = TestGene()
        result = gene.to_dict()

        assert result["timestamp"] == "2023-01-01T12:00:00"

    def test_from_dict_basic(self):
        """基本的な辞書からの復元テスト"""
        class TestGene(BaseGene):
            def __init__(self, name: str, value: int):
                self.name = name
                self.value = value

            def _validate_parameters(self, errors):
                pass

        data = {"name": "test", "value": 123}
        gene = TestGene.from_dict(data)

        assert gene.name == "test"
        assert gene.value == 123

    def test_from_dict_with_enum(self):
        """Enumを含む辞書からの復元テスト"""
        from enum import Enum

        class TestEnum(Enum):
            VALUE1 = "value1"

        class TestGene(BaseGene):
            def __init__(self, enum_attr: TestEnum):
                self.enum_attr = enum_attr

            def _validate_parameters(self, errors):
                pass

        data = {"enum_attr": "value1"}
        gene = TestGene.from_dict(data)

        assert gene.enum_attr == TestEnum.VALUE1

    def test_from_dict_with_datetime(self):
        """datetimeを含む辞書からの復元テスト"""
        class TestGene(BaseGene):
            def __init__(self, timestamp: datetime):
                self.timestamp = timestamp

            def _validate_parameters(self, errors):
                pass

        data = {"timestamp": "2023-01-01T12:00:00"}
        gene = TestGene.from_dict(data)

        assert gene.timestamp == datetime(2023, 1, 1, 12, 0, 0)

    def test_validate_success(self):
        """正常な検証テスト"""
        class TestGene(BaseGene):
            def __init__(self):
                self.enabled = True

            def _validate_parameters(self, errors):
                pass

        gene = TestGene()
        is_valid, errors = gene.validate()

        assert is_valid is True
        assert errors == []

    def test_validate_with_errors(self):
        """エラーありの検証テスト"""
        class TestGene(BaseGene):
            def __init__(self):
                self.enabled = "not_bool"

            def _validate_parameters(self, errors):
                errors.append("custom error")

        gene = TestGene()
        is_valid, errors = gene.validate()

        assert is_valid is False
        assert len(errors) >= 2
        assert "enabled属性がbool型である必要があります" in errors
        assert "custom error" in errors

    def test_validate_range_valid(self):
        """範囲検証で正常な場合"""
        class TestGene(BaseGene):
            def _validate_parameters(self, errors):
                self._validate_range(50, 0, 100, "test_param", errors)

        gene = TestGene()
        is_valid, errors = gene.validate()

        assert is_valid is True

    def test_validate_range_invalid(self):
        """範囲検証で範囲外の場合"""
        class TestGene(BaseGene):
            def _validate_parameters(self, errors):
                result = self._validate_range(150, 0, 100, "test_param", errors)
                assert result is False

        gene = TestGene()
        is_valid, errors = gene.validate()

        assert is_valid is False
        assert "test_paramは0-100の範囲である必要があります" in errors


class TestGeneticUtils:
    """GeneticUtilsクラスのテスト"""

    def test_create_child_metadata(self):
        """子メタデータの作成テスト"""
        parent1_meta = {"fitness": 0.8, "age": 1}
        parent2_meta = {"fitness": 0.7, "age": 2}

        child1_meta, child2_meta = GeneticUtils.create_child_metadata(
            parent1_meta, parent2_meta, "parent1_id", "parent2_id"
        )

        assert child1_meta["fitness"] == 0.8
        assert child1_meta["age"] == 1
        assert child1_meta["crossover_parent1"] == "parent1_id"
        assert child1_meta["crossover_parent2"] == "parent2_id"

        assert child2_meta["fitness"] == 0.7
        assert child2_meta["age"] == 2
        assert child2_meta["crossover_parent1"] == "parent1_id"
        assert child2_meta["crossover_parent2"] == "parent2_id"

    def test_crossover_generic_genes_basic(self):
        """基本的な遺伝子交叉テスト"""
        class MockGene:
            def __init__(self, value1, value2):
                self.__dict__.update({"value1": value1, "value2": value2})

        parent1 = MockGene(10, "A")
        parent2 = MockGene(20, "B")

        with patch('random.random', side_effect=[0.3, 0.6]):
            with patch('random.choice', side_effect=["A", "A"]):
                child1, child2 = GeneticUtils.crossover_generic_genes(
                    parent1, parent2, MockGene,
                    numeric_fields=["value1"],
                    choice_fields=["value2"]
                )

        assert isinstance(child1, MockGene)
        assert isinstance(child2, MockGene)
        assert child1.value1 == (10 + 20) / 2  # 数値フィールドは平均化

    def test_crossover_generic_genes_with_enum(self):
        """Enumフィールドを含む交叉テスト"""
        from enum import Enum

        class TestEnum(Enum):
            VALUE1 = "val1"
            VALUE2 = "val2"

        class MockGene:
            def __init__(self, enum_field):
                self.enum_field = enum_field

        parent1 = MockGene(TestEnum.VALUE1)
        parent2 = MockGene(TestEnum.VALUE2)

        with patch('random.random', side_effect=[0.3]):
            with patch('random.choice', return_value=TestEnum.VALUE1):
                child1, child2 = GeneticUtils.crossover_generic_genes(
                    parent1, parent2, MockGene,
                    enum_fields=["enum_field"]
                )

        assert isinstance(child1.enum_field, TestEnum)
        assert isinstance(child2.enum_field, TestEnum)

    def test_mutate_generic_gene_numeric(self):
        """数値フィールドの突然変異テスト"""
        class MockGene:
            def __init__(self, numeric_field):
                self.numeric_field = numeric_field

        gene = MockGene(50.0)

        with patch('random.random', side_effect=[0.05]):  # mutation happens
            with patch('random.uniform', return_value=1.1):  # mutate factor
                mutated = GeneticUtils.mutate_generic_gene(
                    gene, MockGene,
                    mutation_rate=0.1,
                    numeric_fields=["numeric_field"],
                    numeric_ranges={"numeric_field": (0, 100)}
                )

        assert isinstance(mutated, MockGene)
        # mutation factorの影響を受けるが、範囲内に制限されるはず

    def test_mutate_generic_gene_enum(self):
        """Enumフィールドの突然変異テスト"""
        from enum import Enum

        class TestEnum(Enum):
            VALUE1 = "val1"
            VALUE2 = "val2"

        class MockGene:
            def __init__(self, enum_field):
                self.enum_field = enum_field

        gene = MockGene(TestEnum.VALUE1)

        with patch('random.random', side_effect=[0.05]):  # mutation happens
            with patch('random.choice', return_value=TestEnum.VALUE2):
                mutated = GeneticUtils.mutate_generic_gene(
                    gene, MockGene,
                    mutation_rate=0.1,
                    enum_fields=["enum_field"]
                )

        assert isinstance(mutated, MockGene)
        assert mutated.enum_field == TestEnum.VALUE2


class TestGeneUtils:
    """GeneUtilsクラスのテスト"""

    def test_normalize_parameter_valid(self):
        """有効なパラメータ正規化テスト"""
        result = GeneUtils.normalize_parameter(50, 0, 100)
        assert result == 0.5

        result = GeneUtils.normalize_parameter(100, 0, 100)
        assert result == 1.0

        result = GeneUtils.normalize_parameter(0, 0, 100)
        assert result == 0.0

    def test_normalize_parameter_out_of_range(self):
        """範囲外のパラメータ正規化テスト"""
        # 最小値以下
        result = GeneUtils.normalize_parameter(-10, 0, 100)
        assert result == 0.0

        # 最大値以上
        result = GeneUtils.normalize_parameter(150, 0, 100)
        assert result == 1.0

    def test_create_default_strategy_gene_success(self):
        """デフォルト戦略遺伝子作成の成功テスト"""
        # StrategyGeneクラスをモック
        mock_strategy_gene = Mock()
        mock_strategy_gene.return_value = Mock(id="mock_id")

        with patch('app.services.auto_strategy.utils.gene_utils.uuid') as mock_uuid:
            mock_uuid.uuid4.return_value = "test_uuid"

            # モデルクラスをモック
            with patch.multiple('app.services.auto_strategy.utils.gene_utils',
                              IndicatorGene=Mock,
                              Condition=Mock,
                              TPSLGene=Mock,
                              PositionSizingGene=Mock,
                              PositionSizingMethod=Mock):
                with patch('app.services.auto_strategy.models.strategy_models', create=True) as mock_models:
                    # 実際の実装ではこれらのインポートが必要
                    result = GeneUtils.create_default_strategy_gene(mock_strategy_gene)

        assert result is not None

    def test_create_default_strategy_gene_error(self):
        """デフォルト戦略遺伝子作成のエラー処理テスト"""
        with pytest.raises(ValueError) as exc_info:
            # インポート失敗をシミュレート
            GeneUtils.create_default_strategy_gene("invalid_class")

        assert "デフォルト戦略遺伝子の作成に失敗" in str(exc_info.value)