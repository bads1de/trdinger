import pytest
from backend.app.services.auto_strategy.models.pure_strategy_models import PureIndicatorGene

class TestPureIndicatorGene:
    def test_init_default(self):
        gene = PureIndicatorGene(type="rsi")

        assert gene.type == "rsi"
        assert gene.parameters == {}
        assert gene.enabled is True
        assert gene.json_config == {}

    def test_init_with_parameters(self):
        parameters = {"period": 14, "level": 70}
        gene = PureIndicatorGene(
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
        gene = PureIndicatorGene(type="sma", enabled=True)

        assert gene.enabled is True

    def test_init_with_enabled_false(self):
        gene = PureIndicatorGene(type="sma", enabled=False)

        assert gene.enabled is False