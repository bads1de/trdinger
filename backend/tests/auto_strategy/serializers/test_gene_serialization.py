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
                "app.services.auto_strategy.serializers.list_encoder.ListEncoder"
            ) as mock_list_enc,
            patch(
                "app.services.auto_strategy.serializers.list_decoder.ListDecoder"
            ) as mock_list_dec,
            patch(
                "app.services.auto_strategy.serializers.json_converter.JsonConverter"
            ) as mock_json,
        ):

            serializer = GeneSerializer(enable_smart_generation=True)

            mock_dict.assert_called_once_with(True)
            mock_list_enc.assert_called_once()
            mock_list_dec.assert_called_once_with(True)
            mock_json.assert_called_once()


class TestDelegation:
    """委譲メソッドのテスト"""

    @pytest.fixture
    def serializer(self):
        with (
            patch(
                "app.services.auto_strategy.serializers.dict_converter.DictConverter"
            ),
            patch("app.services.auto_strategy.serializers.list_encoder.ListEncoder"),
            patch("app.services.auto_strategy.serializers.list_decoder.ListDecoder"),
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

    def test_to_list_delegates(self, serializer):
        """to_listの委譲"""
        gene = Mock()
        serializer.to_list(gene)
        serializer.list_encoder.to_list.assert_called_once_with(gene)

    def test_from_list_delegates(self, serializer):
        """from_listの委譲"""
        encoded = [1.0, 2.0]
        cls = Mock()
        serializer.from_list(encoded, cls)
        serializer.list_decoder.from_list.assert_called_once_with(encoded, cls)

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




