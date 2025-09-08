import pytest
from backend.app.services.auto_strategy.models.pure_strategy_models import PureStrategyGene, PositionSizingMethod

class TestPureStrategyGene:
    def test_init_default(self):
        gene = PureStrategyGene()

        assert gene.id == ""
        assert gene.indicators == []
        assert gene.entry_conditions == []
        assert gene.exit_conditions == []
        assert gene.long_entry_conditions == []
        assert gene.short_entry_conditions == []
        assert gene.risk_management == {}
        assert gene.tpsl_gene is None
        assert gene.position_sizing_gene is None
        assert gene.metadata == {}

    def test_init_with_values(self):
        gene = PureStrategyGene(id="test_strategy")

        assert gene.id == "test_strategy"
        assert gene.indicators == []
        assert gene.entry_conditions == []

    def test_get_effective_long_conditions_no_long_conditions(self):
        gene = PureStrategyGene()

        effective = gene.get_effective_long_conditions()

        assert effective == []

    def test_get_effective_short_conditions_no_short_conditions(self):
        gene = PureStrategyGene()

        effective = gene.get_effective_short_conditions()

        assert effective == []

    def test_has_long_short_separation_false(self):
        gene = PureStrategyGene()

        assert gene.has_long_short_separation() is False

    def test_method_property(self):
        gene = PureStrategyGene()

        # Default method when no position_sizing_gene
        assert gene.method == PositionSizingMethod.FIXED_RATIO