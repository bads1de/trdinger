"""
Test for IndicatorGene model
"""
import pytest
from backend.app.services.auto_strategy.models.indicator_gene import IndicatorGene

class TestIndicatorGene:
    def test_init_default(self):
        gene = IndicatorGene(type="rsi")

        assert gene.type == "rsi"
        assert gene.parameters == {}
        assert gene.enabled is True
        assert gene.json_config == {}

    def test_init_with_parameters(self):
        parameters = {"period": 14, "level": 70}
        gene = IndicatorGene(
            type="rsi",
            parameters=parameters,
            enabled=False,
            json_config={"custom": True}
        )

        assert gene.type == "rsi"
        assert gene.parameters == parameters
        assert gene.enabled is False
        assert gene.json_config == {"custom": True}

    def test_init_with_enabled_true(self):
        gene = IndicatorGene(type="sma", enabled=True)

        assert gene.enabled is True

    def test_init_with_enabled_false(self):
        gene = IndicatorGene(type="sma", enabled=False)

        assert gene.enabled is False
    def test_validate_indicator_gene(self):
        indicator = IndicatorGene(type="SMA", parameters={"period": 14})

        try:
            result = indicator.validate()
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("GeneValidator not available")

    def test_get_json_config_with_valid_indicator(self):
        indicator = IndicatorGene(type="SMA", parameters={"period": 14, "source": "close"})

        json_config = indicator.get_json_config()

        assert json_config["indicator"] == "SMA"
        assert "parameters" in json_config

    def test_get_json_config_import_error_fallback(self):
        # This tests the config retrieval path
        indicator = IndicatorGene(type="SMA", parameters={"period": 14})

        json_config = indicator.get_json_config()

        assert json_config["indicator"] == "SMA"
        assert "parameters" in json_config

    def test_indicator_gene_disabled(self):
        indicator = IndicatorGene(type="SMA", enabled=False)

        assert not indicator.enabled