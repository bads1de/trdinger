import pytest
from backend.app.services.auto_strategy.models.pure_strategy_models import PureTPSLGene, TPSLMethod

class TestPureTPSLGene:
    def test_init_default(self):
        gene = PureTPSLGene()

        assert gene.method == TPSLMethod.RISK_REWARD_RATIO
        assert gene.stop_loss_pct == 0.03
        assert gene.take_profit_pct == 0.06
        assert gene.risk_reward_ratio == 2.0
        assert gene.base_stop_loss == 0.03
        assert gene.atr_multiplier_sl == 2.0
        assert gene.atr_multiplier_tp == 3.0
        assert gene.atr_period == 14
        assert gene.lookback_period == 100
        assert gene.confidence_threshold == 0.7
        assert gene.enabled is True
        assert gene.priority == 1.0
        assert gene.method_weights == {
            "fixed": 0.25,
            "risk_reward": 0.35,
            "volatility": 0.25,
            "statistical": 0.15,
        }

    def test_init_with_values(self):
        gene = PureTPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE,
            stop_loss_pct=0.05,
            enabled=False
        )

        assert gene.method == TPSLMethod.FIXED_PERCENTAGE
        assert gene.stop_loss_pct == 0.05
        assert gene.enabled is False

    def test_init_with_custom_method_weights(self):
        custom_weights = {
            "fixed": 0.5,
            "risk_reward": 0.3,
            "volatility": 0.1,
            "statistical": 0.1,
        }
        gene = PureTPSLGene(method_weights=custom_weights)

        assert gene.method_weights == custom_weights