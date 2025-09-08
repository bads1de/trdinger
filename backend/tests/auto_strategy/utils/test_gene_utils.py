"""
遺伝子関連ユーティリティのユニットテスト

TDDアプローチで開発されたgene_utils.pyのテストケースを提供します。
"""

import pytest
from unittest.mock import Mock
from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.app.services.auto_strategy.utils.gene_utils import (
    BaseGene,
    GeneUtils,
    GeneticUtils,
    normalize_parameter,
    create_default_strategy_gene,
    create_child_metadata,
    prepare_crossover_metadata,
)


# ============================================================================
# Mock Classes for Testing
# ============================================================================


class MockGene(BaseGene):
    """テスト用のMock遺伝子クラス"""

    def __init__(
        self,
        id: Optional[str] = None,
        enabled: bool = True,
        value: int = 10,
        name: str = "test_gene",
        created_at: Optional[datetime] = None,
    ):
        self.id = id
        self.enabled = enabled
        self.value = value
        self.name = name
        self.created_at = created_at or datetime.now()

    def _validate_parameters(self, errors: List[str]) -> None:
        if self.value < 0 or self.value > 100:
            errors.append("valueは0-100の範囲である必要があります")
        if not self.name:
            errors.append("nameは必須です")


class MockStrategyGene:
    """テスト用のMock戦略遺伝子"""

    def __init__(self, id: str = "test_id", metadata: Dict[str, Any] = None):
        self.id = id
        self.metadata = metadata or {"generation": 1}


# ============================================================================
# BaseGene Tests
# ============================================================================


class TestBaseGene:
    """BaseGeneクラスのテスト"""

    def test_to_dict_basic(self):
        """基本的なto_dictテスト"""
        gene = MockGene(id="123", enabled=True, value=25, name="test")
        result = gene.to_dict()

        assert "id" in result
        assert "enabled" in result
        assert "value" in result
        assert "name" in result
        assert "created_at" in result

        assert result["id"] == "123"
        assert result["enabled"] is True
        assert result["value"] == 25
        assert result["name"] == "test"

    def test_to_dict_with_datetime(self):
        """datetimeを含むto_dictテスト"""
        test_time = datetime(2023, 1, 1, 12, 0, 0)
        gene = MockGene(created_at=test_time)
        result = gene.to_dict()

        assert result["created_at"] == "2023-01-01T12:00:00"

    def test_to_dict_excludes_private_attrs(self):
        """プライベート属性の除外テスト"""
        gene = MockGene(name="test")
        gene._private_attr = "secret"
        result = gene.to_dict()

        assert "_private_attr" not in result

    def test_from_dict_without_annotations(self):
        """アノテーションなしのfrom_dictテスト"""
        data = {"id": "123", "enabled": True, "value": 50, "name": "test"}
        gene = MockGene.from_dict(data)

        assert isinstance(gene, MockGene)
        assert gene.value == 50

    def test_validate_success(self):
        """正常な検証テスト"""
        gene = MockGene(value=50, name="valid")
        is_valid, errors = gene.validate()

        assert is_valid is True
        assert errors == []

    def test_validate_failure(self):
        """検証失敗テスト"""
        gene = MockGene(value=-5, name="")
        is_valid, errors = gene.validate()

        assert is_valid is False
        assert len(errors) == 2
        assert any("value" in error for error in errors)
        assert any("name" in error for error in errors)

    def test_validate_enabled_type(self):
        """enabled属性の型検証テスト"""
        gene = MockGene()
        gene.enabled = "true"  # 文字列なのでbool型でない
        is_valid, errors = gene.validate()

        assert is_valid is False
        assert any("enabled" in error for error in errors)


# ============================================================================
# GeneUtils Tests
# ============================================================================


class TestGeneUtils:
    """GeneUtilsクラスのテスト"""

    def test_normalize_parameter_normal_case(self):
        """標準的なパラメータ正規化テスト"""
        result = GeneUtils.normalize_parameter(50, 0, 100)
        assert result == 0.5

    def test_normalize_parameter_min_max(self):
        """最小値と最大値の境界テスト"""
        assert GeneUtils.normalize_parameter(0, 0, 100) == 0.0
        assert GeneUtils.normalize_parameter(100, 0, 100) == 1.0

    def test_normalize_parameter_out_of_range(self):
        """範囲外の値テスト"""
        assert GeneUtils.normalize_parameter(-10, 0, 100) == 0.0
        assert GeneUtils.normalize_parameter(150, 0, 100) == 1.0

    def test_normalize_parameter_custom_range(self):
        """カスタム範囲テスト"""
        result = GeneUtils.normalize_parameter(150, 100, 200)
        assert result == 0.5

    def test_create_default_strategy_gene(self):
        """デフォルト戦略遺伝子の作成テスト"""
        mock_gene_class = Mock
        gene = GeneUtils.create_default_strategy_gene(mock_gene_class)

        assert gene.mock_name == mock_gene_class
        # 実際のデータ構造はMockなので断定できないが、関数が例外を投げないことを確認


# ============================================================================
# GeneticUtils Tests
# ============================================================================


class TestGeneticUtils:
    """GeneticUtilsクラスのテスト"""

    def test_create_child_metadata(self):
        """子メタデータ作成テスト"""
        parent1_md = {"generation": 1, "fitness": 0.8}
        parent2_md = {"generation": 1, "fitness": 0.7}

        child1_md, child2_md = GeneticUtils.create_child_metadata(
            parent1_md, parent2_md, "parent1_id", "parent2_id"
        )

        assert child1_md["generation"] == 1
        assert child2_md["generation"] == 1
        assert child1_md["crossover_parent1"] == "parent1_id"
        assert child1_md["crossover_parent2"] == "parent2_id"
        assert child2_md["crossover_parent1"] == "parent1_id"
        assert child2_md["crossover_parent2"] == "parent2_id"

    def test_prepare_crossover_metadata(self):
        """交叉メタデータ準備テスト"""
        parent1 = MockStrategyGene("parent1_id", {"gen": 1, "fit": 0.8})
        parent2 = MockStrategyGene("parent2_id", {"gen": 1, "fit": 0.6})

        child1_md, child2_md = GeneticUtils.prepare_crossover_metadata(parent1, parent2)

        assert child1_md["gen"] == 1
        assert child1_md["fit"] == 0.8
        assert child1_md["crossover_parent1"] == "parent1_id"
        assert child1_md["crossover_parent2"] == "parent2_id"

    def test_crossover_generic_genes_basic(self):
        """継承遺伝子の基本テスト"""
        parent1 = MockGene(id="p1", enabled=True, value=10)
        parent2 = MockGene(id="p2", enabled=False, value=20)
        mock_gene_class = MockGene

        child1, child2 = GeneticUtils.crossover_generic_genes(
            parent1, parent2, mock_gene_class, [], [], ["enabled"]
        )

        assert isinstance(child1, MockGene)
        assert isinstance(child2, MockGene)
        assert child1.enabled in [True, False]
        assert child2.enabled in [True, False]

    def test_crossover_numeric_fields(self):
        """数値フィールド平均化テスト"""
        parent1 = MockGene(value=10)
        parent2 = MockGene(value=20)
        mock_gene_class = MockGene

        child1, child2 = GeneticUtils.crossover_generic_genes(
            parent1, parent2, mock_gene_class, ["value"]
        )

        assert child1.value == 15.0
        assert child2.value == 15.0

    def test_crossover_enum_fields(self):
        """Enumフィールドランダム選択テスト"""
        # テストのためenum_likeオブジェクトを作成
        mock_enum = Mock()
        mock_enum.__class__.__members__ = ["A", "B"]
        mock_enum.__class__.__getitem__ = Mock(side_effect=lambda k: k)

        parent1 = MockGene()
        parent2 = MockGene()
        parent1.enabled = mock_enum  # enabledをenum_likeに変更
        parent2.enabled = mock_enum
        mock_gene_class = MockGene

        # enum_fieldsに"enabled"を指定
        child1, child2 = GeneticUtils.crossover_generic_genes(
            parent1, parent2, mock_gene_class, [], ["enabled"]
        )

        # pytestでランダムなので検証は難しいが、例外が発生しないことを確認
        assert isinstance(child1, MockGene)


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """便利関数のテスト"""

    def test_normalize_parameter_global(self):
        """グローバルnormalize_parameter関数のテスト"""
        assert normalize_parameter(50, 0, 100) == 0.5

    def test_create_default_strategy_gene_global(self):
        """グローバルcreate_default_strategy_gene関数のテスト"""
        mock_gene_class = Mock
        gene = create_default_strategy_gene(mock_gene_class)

        assert gene.mock_name == mock_gene_class

    def test_create_child_metadata_global(self):
        """グローバルcreate_child_metadata関数のテスト"""
        parent1_md = {"test": 1}
        parent2_md = {"test": 2}

        child1, child2 = create_child_metadata(parent1_md, parent2_md, "p1", "p2")

        assert child1["crossover_parent1"] == "p1"
        assert child1["crossover_parent2"] == "p2"

if __name__ == "__main__":
    # 直接実行時のpytest実行
    pytest.main([__file__, "-v"])