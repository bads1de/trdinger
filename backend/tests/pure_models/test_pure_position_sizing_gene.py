import pytest
from backend.app.services.auto_strategy.models.pure_strategy_models import PurePositionSizingGene, PositionSizingMethod

class TestPurePositionSizingGene:
    def test_init_default(self):
        gene = PurePositionSizingGene()

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
        gene = PurePositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            risk_per_trade=0.05,
            enabled=False
        )

        assert gene.method == PositionSizingMethod.FIXED_RATIO
        assert gene.risk_per_trade == 0.05
        assert gene.enabled is False

    def test_init_with_negative_values(self):
        gene = PurePositionSizingGene(
            risk_per_trade=-0.02,
            min_position_size=-0.1
        )

        assert gene.risk_per_trade == -0.02
        assert gene.min_position_size == -0.1