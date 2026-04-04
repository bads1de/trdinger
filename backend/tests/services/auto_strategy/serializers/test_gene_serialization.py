"""
GeneSerializerのテスト
"""

import json
import pytest
from unittest.mock import Mock, patch
from app.services.auto_strategy.genes import Condition, IndicatorGene, StrategyGene
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

    def test_from_list_returns_strategy_gene_for_list_like_individual(self, serializer):
        """DEAP個体風のlistからStrategyGeneを復元できること"""
        gene = StrategyGene(
            id="list-like",
            indicators=[IndicatorGene(type="SMA", parameters={"period": 10})],
            metadata={},
        )

        class MockIndividual(list):
            def __init__(self, item):
                super().__init__([item])

        individual = MockIndividual(gene)

        restored = serializer.from_list(individual, StrategyGene)

        assert isinstance(restored, StrategyGene)
        assert restored is gene


class TestGeneSerializerCacheIntegration:
    """キャッシュ統合のテスト"""

    @pytest.fixture
    def serializer(self):
        return GeneSerializer()

    @pytest.fixture
    def gene(self):
        return StrategyGene(
            id="gene-cache-test",
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ],
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="open")
            ],
            short_entry_conditions=[],
            metadata={"nested": {"flag": True}},
        )

    def test_strategy_gene_to_dict_uses_cache_and_returns_copies(
        self,
        serializer,
        gene,
    ):
        first = serializer.strategy_gene_to_dict(gene)
        first["metadata"]["nested"]["flag"] = False

        stats_after_first = serializer.get_cache_statistics()
        assert stats_after_first["serialize_cache_size"] == 1

        second = serializer.strategy_gene_to_dict(gene)
        stats_after_second = serializer.get_cache_statistics()

        assert stats_after_second["serialize_cache_size"] == 1
        assert first is not second
        assert first["metadata"] is not second["metadata"]
        assert second["metadata"]["nested"]["flag"] is True

    def test_dict_to_strategy_gene_uses_cache_and_returns_copies(
        self,
        serializer,
        gene,
    ):
        data = serializer.strategy_gene_to_dict(gene)
        restored_first = serializer.dict_to_strategy_gene(data, StrategyGene)
        restored_first.metadata["nested"]["flag"] = False

        stats_after_first = serializer.get_cache_statistics()
        assert stats_after_first["deserialize_cache_size"] == 1

        restored_second = serializer.dict_to_strategy_gene(data, StrategyGene)
        stats_after_second = serializer.get_cache_statistics()

        assert stats_after_second["deserialize_cache_size"] == 1
        assert restored_first is not restored_second
        assert restored_second.metadata["nested"]["flag"] is True

    def test_strategy_gene_to_dict_distinguishes_other_fields(
        self,
        serializer,
    ):
        same_indicators = [
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
        ]

        first_gene = StrategyGene(
            id="",
            indicators=same_indicators,
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="open")
            ],
            short_entry_conditions=[],
            metadata={"tag": "first"},
        )
        second_gene = StrategyGene(
            id="",
            indicators=same_indicators,
            long_entry_conditions=[
                Condition(left_operand="close", operator="<", right_operand="open")
            ],
            short_entry_conditions=[],
            metadata={"tag": "second"},
        )

        first = serializer.strategy_gene_to_dict(first_gene)
        second = serializer.strategy_gene_to_dict(second_gene)

        assert first["metadata"]["tag"] == "first"
        assert second["metadata"]["tag"] == "second"
        assert serializer.get_cache_statistics()["serialize_cache_size"] == 2
