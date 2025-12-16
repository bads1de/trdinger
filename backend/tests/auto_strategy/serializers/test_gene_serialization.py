"""
GeneSerializerのテスト
"""

import pytest
from unittest.mock import Mock, patch
from app.services.auto_strategy.serializers.gene_serialization import GeneSerializer


class TestGeneSerializerInit:
    """初期化のテスト"""

    def test_init_initializes_components(self):
        """コンポーネントを初期化する"""
        with (
            patch(
                "app.services.auto_strategy.serializers.dict_converter.DictConverter"
            ) as mock_dict,
            patch(
                "app.services.auto_strategy.serializers.json_converter.JsonConverter"
            ) as mock_json,
        ):

            serializer = GeneSerializer(enable_smart_generation=True)

            mock_dict.assert_called_once_with(True)
            mock_json.assert_called_once()


class TestDelegation:
    """委譲メソッドのテスト"""

    @pytest.fixture
    def serializer(self):
        with (
            patch(
                "app.services.auto_strategy.serializers.dict_converter.DictConverter"
            ),
            patch(
                "app.services.auto_strategy.serializers.json_converter.JsonConverter"
            ),
        ):
            return GeneSerializer()

    def test_strategy_gene_to_dict_delegates(self, serializer):
        """strategy_gene_to_dictの委譲"""
        gene = Mock()
        serializer.strategy_gene_to_dict(gene)
        serializer.dict_converter.strategy_gene_to_dict.assert_called_once_with(gene)

    def test_json_conversion_delegates(self, serializer):
        """JSON変換の委譲"""
        gene = Mock()
        json_str = "{}"
        cls = Mock()

        serializer.strategy_gene_to_json(gene)
        serializer.json_converter.strategy_gene_to_json.assert_called_once_with(gene)

        serializer.json_to_strategy_gene(json_str, cls)
        serializer.json_converter.json_to_strategy_gene.assert_called_once_with(
            json_str, cls
        )





