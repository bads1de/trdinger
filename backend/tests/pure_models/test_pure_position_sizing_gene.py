"""
Test for PositionSizingGene model
"""
import pytest
from backend.app.services.auto_strategy.models.position_sizing_gene import PositionSizingGene
from backend.app.services.auto_strategy.models.enums import PositionSizingMethod

class TestPositionSizingGene:
    def test_init_default(self):
        gene = PositionSizingGene()

        assert gene.method == PositionSizingMethod.VOLATILITY_BASED
        assert gene.lookback_period == 100
        assert gene.optimal_f_multiplier == 0.5
        assert gene.atr_period == 14
        assert gene.atr_multiplier == 2.0
        assert gene.risk_per_trade == 0.02
        assert gene.fixed_ratio == 0.1
        assert gene.fixed_quantity == 1.0
        assert gene.min_position_size == 0.01
        assert gene.max_position_size == 9999.0
        assert gene.enabled is True
        assert gene.priority == 1.0

    def test_init_with_values(self):
        gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            risk_per_trade=0.05,
            enabled=False
        )

        assert gene.method == PositionSizingMethod.FIXED_RATIO
        assert gene.risk_per_trade == 0.05
        assert gene.enabled is False

    def test_init_with_negative_values(self):
        gene = PositionSizingGene(
            risk_per_trade=-0.02,
            min_position_size=-0.1
        )

        assert gene.risk_per_trade == -0.02
        assert gene.min_position_size == -0.1
    def test_validate_parameters_valid(self):
        gene = PositionSizingGene(
            lookback_period=150,
            risk_per_trade=0.02,
            fixed_ratio=0.1,
            atr_multiplier=2.0
        )

        errors = []
        gene._validate_parameters(errors)

        assert len(errors) == 0

    def test_validate_parameters_invalid_lookback_period(self):
        gene = PositionSizingGene(lookback_period=1000)

        errors = []
        gene._validate_parameters(errors)

        assert len(errors) > 0
        assert "lookback_period" in errors[0]

    def test_validate_parameters_invalid_risk_per_trade(self):
        gene = PositionSizingGene(risk_per_trade=0.15)

        errors = []
        gene._validate_parameters(errors)

        assert len(errors) > 0
        assert "risk_per_trade" in errors[0]

    def test_validate_parameters_invalid_fixed_ratio(self):
        gene = PositionSizingGene(fixed_ratio=1.5)

        errors = []
        gene._validate_parameters(errors)

        assert len(errors) > 0
        assert "fixed_ratio" in errors[0]