import pytest

from app.services.auto_strategy.genes import EntryGene
from app.services.auto_strategy.serializers.strategy_gene_dict_codec import (
    StrategyGeneDictCodec,
)


class _DummyConverter:
    pass


@pytest.fixture
def codec() -> StrategyGeneDictCodec:
    return StrategyGeneDictCodec(_DummyConverter())


def test_get_sub_gene_field_names_accepts_string_iterable_override(
    codec: StrategyGeneDictCodec,
) -> None:
    class CustomStrategyGene:
        @staticmethod
        def sub_gene_field_names() -> list[str]:
            return ["custom_entry_gene", "custom_exit_gene"]

    assert codec._get_sub_gene_field_names(CustomStrategyGene) == (
        "custom_entry_gene",
        "custom_exit_gene",
    )


def test_get_sub_gene_field_names_rejects_scalar_string_override(
    codec: StrategyGeneDictCodec,
) -> None:
    class InvalidStrategyGene:
        @staticmethod
        def sub_gene_field_names() -> str:
            return "custom_entry_gene"

    with pytest.raises(TypeError, match="sub_gene_field_names"):
        codec._get_sub_gene_field_names(InvalidStrategyGene)


def test_get_sub_gene_class_map_accepts_mapping_override(
    codec: StrategyGeneDictCodec,
) -> None:
    class CustomStrategyGene:
        @staticmethod
        def sub_gene_class_map() -> dict[str, type[EntryGene]]:
            return {"custom_entry_gene": EntryGene}

    assert codec._get_sub_gene_class_map(CustomStrategyGene) == {
        "custom_entry_gene": EntryGene
    }


def test_get_sub_gene_class_map_rejects_non_string_keys(
    codec: StrategyGeneDictCodec,
) -> None:
    class InvalidStrategyGene:
        @staticmethod
        def sub_gene_class_map() -> dict[bytes, type[EntryGene]]:
            return {b"custom_entry_gene": EntryGene}

    with pytest.raises(TypeError, match="sub_gene_class_map"):
        codec._get_sub_gene_class_map(InvalidStrategyGene)
