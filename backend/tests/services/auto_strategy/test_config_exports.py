import importlib

import pytest

from app.services.auto_strategy import config as auto_strategy_config


def test_config_public_exports_are_trimmed():
    assert hasattr(auto_strategy_config, "BaseConfig") is False
    assert not hasattr(auto_strategy_config, "AutoStrategyConfig")
    assert auto_strategy_config.GAConfig is not None
    assert not hasattr(auto_strategy_config, "TradingSettings")
    assert not hasattr(auto_strategy_config, "IndicatorSettings")
    assert not hasattr(auto_strategy_config, "TPSLSettings")
    assert not hasattr(auto_strategy_config, "PositionSizingSettings")


def test_legacy_auto_strategy_settings_module_is_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(
            "app.services.auto_strategy.config.auto_strategy_settings"
        )


def test_legacy_settings_module_is_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("app.services.auto_strategy.config.settings")


def test_legacy_base_module_is_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("app.services.auto_strategy.config.base")
