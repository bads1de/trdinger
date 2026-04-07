from types import SimpleNamespace

from app.services.auto_strategy.config.ga import GAConfig
from app.services.auto_strategy.config.sub_configs import (
    EarlyTerminationSettings,
    EvaluationConfig,
    resolve_early_termination_settings,
)
from app.services.auto_strategy.config.validators import ConfigValidator


class TestEarlyTerminationSettings:
    def test_resolve_settings_from_legacy_flat_fields(self):
        source = SimpleNamespace(
            enable_early_termination=True,
            early_termination_max_drawdown=0.2,
            early_termination_min_trades=12,
            early_termination_min_trade_check_progress=0.45,
            early_termination_trade_pace_tolerance=0.6,
            early_termination_min_expectancy=-0.02,
            early_termination_expectancy_min_trades=6,
            early_termination_expectancy_progress=0.7,
        )

        settings = resolve_early_termination_settings(source)

        assert settings == EarlyTerminationSettings(
            enabled=True,
            max_drawdown=0.2,
            min_trades=12,
            min_trade_check_progress=0.45,
            trade_pace_tolerance=0.6,
            min_expectancy=-0.02,
            expectancy_min_trades=6,
            expectancy_progress=0.7,
        )
        assert settings.to_strategy_params() == {
            "enable_early_termination": True,
            "early_termination_max_drawdown": 0.2,
            "early_termination_min_trades": 12,
            "early_termination_min_trade_check_progress": 0.45,
            "early_termination_trade_pace_tolerance": 0.6,
            "early_termination_min_expectancy": -0.02,
            "early_termination_expectancy_min_trades": 6,
            "early_termination_expectancy_progress": 0.7,
        }

    def test_resolve_settings_prefers_nested_settings(self):
        source = {
            "early_termination_settings": {
                "enabled": True,
                "max_drawdown": 0.15,
                "min_trades": 20,
                "min_trade_check_progress": 0.4,
                "trade_pace_tolerance": 0.5,
                "min_expectancy": -0.01,
                "expectancy_min_trades": 5,
                "expectancy_progress": 0.6,
            },
            "enable_early_termination": False,
            "early_termination_max_drawdown": 0.9,
        }

        settings = resolve_early_termination_settings(source)

        assert settings.enabled is True
        assert settings.max_drawdown == 0.15
        assert settings.min_trades == 20
        assert settings.to_strategy_params()["enable_early_termination"] is True

    def test_ga_config_from_dict_syncs_nested_settings(self):
        restored = GAConfig.from_dict(
            {
                "early_termination_settings": {
                    "enabled": True,
                    "max_drawdown": 0.15,
                    "min_trades": 20,
                    "min_trade_check_progress": 0.4,
                    "trade_pace_tolerance": 0.5,
                    "min_expectancy": -0.01,
                    "expectancy_min_trades": 5,
                    "expectancy_progress": 0.6,
                },
                "enable_early_termination": False,
                "early_termination_max_drawdown": 0.9,
            }
        )

        assert restored.early_termination_settings is not None
        assert restored.early_termination_settings.enabled is True
        assert restored.early_termination_settings.max_drawdown == 0.15
        assert restored.enable_early_termination is True
        assert restored.early_termination_max_drawdown == 0.15
        assert restored.early_termination_expectancy_progress == 0.6

    def test_ga_config_direct_init_syncs_nested_settings(self):
        config = GAConfig(
            early_termination_settings=EarlyTerminationSettings(
                enabled=True,
                max_drawdown=0.12,
                min_trades=8,
                min_trade_check_progress=0.55,
                trade_pace_tolerance=0.65,
                min_expectancy=-0.03,
                expectancy_min_trades=4,
                expectancy_progress=0.75,
            ),
            enable_early_termination=False,
            early_termination_max_drawdown=0.9,
            early_termination_min_trades=99,
        )

        assert config.early_termination_settings is not None
        assert config.early_termination_settings.enabled is True
        assert config.enable_early_termination is True
        assert config.early_termination_max_drawdown == 0.12
        assert config.early_termination_min_trades == 8
        assert config.early_termination_expectancy_progress == 0.75

    def test_config_validator_uses_nested_settings(self):
        config = GAConfig.from_dict(
            {
                "early_termination_settings": {
                    "enabled": True,
                    "max_drawdown": 1.5,
                    "expectancy_min_trades": 0,
                }
            }
        )

        is_valid, errors = ConfigValidator.validate(config)

        assert is_valid is False
        assert any("early_termination_max_drawdown" in e for e in errors)
        assert any("early_termination_expectancy_min_trades" in e for e in errors)

    def test_evaluation_config_from_dict_accepts_legacy_flat_fields(self):
        restored = EvaluationConfig.from_dict(
            {
                "enable_early_termination": True,
                "early_termination_max_drawdown": 0.2,
                "early_termination_min_trades": 12,
                "early_termination_min_trade_check_progress": 0.45,
                "early_termination_trade_pace_tolerance": 0.6,
                "early_termination_min_expectancy": -0.02,
                "early_termination_expectancy_min_trades": 6,
                "early_termination_expectancy_progress": 0.7,
            }
        )

        assert restored.early_termination_settings.enabled is True
        assert restored.early_termination_settings.max_drawdown == 0.2
        assert restored.early_termination_settings.expectancy_min_trades == 6
