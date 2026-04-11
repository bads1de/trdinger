from app.services.auto_strategy.config.ga import GAConfig
from app.services.auto_strategy.config.ga.nested_configs import (
    EarlyTerminationSettings,
    EvaluationConfig,
)
from app.services.auto_strategy.config import ConfigValidator


class TestEarlyTerminationSettings:
    def test_from_dict_accepts_nested_settings(self):
        settings = EarlyTerminationSettings.from_dict(
            {
                "enabled": True,
                "max_drawdown": 0.15,
                "min_trades": 20,
                "min_trade_check_progress": 0.4,
                "trade_pace_tolerance": 0.5,
                "min_expectancy": -0.01,
                "expectancy_min_trades": 5,
                "expectancy_progress": 0.6,
            }
        )

        assert settings.enabled is True
        assert settings.max_drawdown == 0.15
        assert settings.min_trades == 20

    def test_evaluation_config_uses_nested_settings(self):
        config = EvaluationConfig.from_dict(
            {
                "early_termination_settings": {
                    "enabled": True,
                    "max_drawdown": 0.15,
                    "expectancy_min_trades": 5,
                }
            }
        )

        assert config.early_termination_settings.enabled is True
        assert config.early_termination_settings.max_drawdown == 0.15
        assert config.early_termination_settings.expectancy_min_trades == 5

    def test_config_validator_uses_nested_settings(self):
        config = GAConfig.from_dict(
            {
                "evaluation_config": {
                    "early_termination_settings": {
                        "enabled": True,
                        "max_drawdown": 1.5,
                        "expectancy_min_trades": 0,
                    }
                }
            }
        )

        is_valid, errors = ConfigValidator.validate(config)

        assert is_valid is False
        assert any(
            "evaluation_config.early_termination_settings.max_drawdown" in e
            for e in errors
        )
        assert any(
            "evaluation_config.early_termination_settings.expectancy_min_trades" in e
            for e in errors
        )
