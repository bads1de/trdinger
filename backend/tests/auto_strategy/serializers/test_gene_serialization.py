"""
GeneSerializerのテスト
"""

import json
import pytest
from unittest.mock import Mock, patch
from app.services.auto_strategy.serializers.serialization import GeneSerializer


class TestGeneSerializerInit:
    """初期化のテスト"""

    def test_init_initializes_components(self):
        """コンポーネントを初期化する"""
        with patch(
            "app.services.auto_strategy.serializers.serialization.DictConverter"
        ) as mock_dict:
            serializer = GeneSerializer(enable_smart_generation=True)
            mock_dict.assert_called_once_with(True)


class TestDelegation:
    """委譲メソッドのテスト"""

    @pytest.fixture
    def serializer(self):
        with patch(
            "app.services.auto_strategy.serializers.serialization.DictConverter"
        ):
            return GeneSerializer()

    def test_strategy_gene_to_dict_delegates(self, serializer):
        """strategy_gene_to_dictの委譲"""
        gene = Mock()
        serializer.strategy_gene_to_dict(gene)
        serializer.dict_converter.strategy_gene_to_dict.assert_called_once_with(gene)

    def test_json_conversion(self, serializer):
        """JSON変換のテスト（委譲ではなく統合されたロジック）"""
        gene = Mock()
        expected_dict = {"id": "test"}
        serializer.dict_converter.strategy_gene_to_dict.return_value = expected_dict
        
        # JSON変換
        json_str = serializer.strategy_gene_to_json(gene)
        assert json.loads(json_str) == expected_dict
        serializer.dict_converter.strategy_gene_to_dict.assert_called_once_with(gene)

        # JSON復元
        cls = Mock()
        serializer.json_to_strategy_gene(json_str, cls)
        serializer.dict_converter.dict_to_strategy_gene.assert_called_once_with(expected_dict, cls)