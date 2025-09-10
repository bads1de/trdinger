"""
Test for enums
"""
import pytest
from backend.app.services.auto_strategy.models.enums import PositionSizingMethod, TPSLMethod


class TestEnums:
    def test_position_sizing_method_values(self):
        assert PositionSizingMethod.HALF_OPTIMAL_F.value == "half_optimal_f"
        assert PositionSizingMethod.VOLATILITY_BASED.value == "volatility_based"
        assert PositionSizingMethod.FIXED_RATIO.value == "fixed_ratio"
        assert PositionSizingMethod.FIXED_QUANTITY.value == "fixed_quantity"

    def test_tpsl_method_values(self):
        assert TPSLMethod.FIXED_PERCENTAGE.value == "fixed_percentage"
        assert TPSLMethod.RISK_REWARD_RATIO.value == "risk_reward_ratio"
        assert TPSLMethod.VOLATILITY_BASED.value == "volatility_based"
        assert TPSLMethod.STATISTICAL.value == "statistical"
        assert TPSLMethod.ADAPTIVE.value == "adaptive"

    def test_enum_iteration(self):
        ps_methods = list(PositionSizingMethod)
        assert len(ps_methods) == 4

        tpsl_methods = list(TPSLMethod)
        assert len(tpsl_methods) == 5