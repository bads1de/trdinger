"""strategies パッケージの lazy import を検証する。"""

from __future__ import annotations

import importlib
import sys


class TestStrategiesPackageLazyImports:
    def test_strategies_package_defers_universal_strategy_import(self) -> None:
        sys.modules.pop("app.services.auto_strategy.strategies", None)
        sys.modules.pop("app.services.auto_strategy.strategies.universal_strategy", None)

        module = importlib.import_module("app.services.auto_strategy.strategies")

        assert "UniversalStrategy" not in module.__dict__
        assert "app.services.auto_strategy.strategies.universal_strategy" not in sys.modules
