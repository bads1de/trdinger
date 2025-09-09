import pytest
from app.services.auto_strategy.services.auto_strategy_service import (
    AutoStrategyService,
)


def test_build_ga_config_method_exists():
    svc = AutoStrategyService(enable_smart_generation=False)
    assert hasattr(svc, "_build_ga_config_from_dict") or hasattr(
        svc, "build_ga_config"
    ), "AutoStrategyService should expose a method to build GA config from dict"
