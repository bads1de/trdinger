"""
GeneSerializerのテスト
"""

import json
import pytest
from unittest.mock import Mock, patch
from app.services.auto_strategy.serializers.serialization import GeneSerializer


class TestGeneSerializerInit:
    """初期化のテスト"""

    def test_init_exposes_self_as_converter(self):
        """後方互換のため dict_converter は self を指す"""
        serializer = GeneSerializer()
        assert serializer.dict_converter is serializer


class TestDelegation:
    """JSON 補助メソッドのテスト"""

    @pytest.fixture
    def serializer(self):
        return GeneSerializer()

    def test_json_conversion(self, serializer):
        """JSON変換と復元が継承メソッド経由で動くこと"""
        gene = Mock()
        expected_dict = {"id": "test"}
        restored_gene = Mock()

        with patch.object(
            serializer,
            "strategy_gene_to_dict",
            return_value=expected_dict,
        ) as mock_to_dict:
            json_str = serializer.strategy_gene_to_json(gene)
            assert json.loads(json_str) == expected_dict
            mock_to_dict.assert_called_once_with(gene)

        cls = Mock()
        with patch.object(
            serializer,
            "dict_to_strategy_gene",
            return_value=restored_gene,
        ) as mock_from_dict:
            result = serializer.json_to_strategy_gene(json_str, cls)
            assert result is restored_gene
            mock_from_dict.assert_called_once_with(expected_dict, cls)
