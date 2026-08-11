"""auto_strategy パッケージの lazy import を検証する。"""

from __future__ import annotations

import importlib
import sys

import pytest


class TestAutoStrategyPackageLazyImports:
    """循環依存を避けるための lazy import を検証する。"""

    def test_auto_strategy_package_defers_position_sizing_import(self) -> None:
        """auto_strategy パッケージは属性アクセスまで PositionSizingService を読み込まない。"""
        sys.modules.pop("app.services.auto_strategy", None)

        module = importlib.import_module("app.services.auto_strategy")

        assert "PositionSizingService" not in module.__dict__

        exported = module.PositionSizingService

        assert exported.__name__ == "PositionSizingService"
        assert module.__dict__["PositionSizingService"] is exported

    def test_services_package_defers_auto_strategy_service_import(self) -> None:
        """services パッケージは属性アクセスまで AutoStrategyService を読み込まない。"""
        sys.modules.pop("app.services.auto_strategy.services", None)

        module = importlib.import_module("app.services.auto_strategy.services")

        assert "AutoStrategyService" not in module.__dict__

        exported = module.AutoStrategyService

        assert exported.__name__ == "AutoStrategyService"
        assert module.__dict__["AutoStrategyService"] is exported

    def test_core_package_defers_evolution_runner_import(self) -> None:
        """core パッケージは属性アクセスまで EvolutionRunner を読み込まない。"""
        sys.modules.pop("app.services.auto_strategy.core", None)

        module = importlib.import_module("app.services.auto_strategy.core")

        assert "EvolutionRunner" not in module.__dict__

        exported = module.EvolutionRunner

        assert exported.__name__ == "EvolutionRunner"
        assert module.__dict__["EvolutionRunner"] is exported

    def test_hybrid_package_is_removed(self) -> None:
        """hybrid パッケージは ML 機能削除に伴い削除済み。"""
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("app.services.auto_strategy.core.hybrid")

    def test_positions_package_defers_position_sizing_import(self) -> None:
        """positions パッケージは属性アクセスまで個別モジュールを読み込まない。"""
        sys.modules.pop("app.services.auto_strategy.positions", None)

        module = importlib.import_module("app.services.auto_strategy.positions")

        assert "PositionSizingService" not in module.__dict__

        exported = module.PositionSizingService

        assert exported.__name__ == "PositionSizingService"
        assert module.__dict__["PositionSizingService"] is exported
        # 他のモジュールはまだ読み込まれていない（個別遅延）
        assert "EntryExecutor" not in module.__dict__

    def test_calculators_package_defers_calculator_import(self) -> None:
        """calculators パッケージは属性アクセスまで個別計算機を読み込まない。"""
        sys.modules.pop("app.services.auto_strategy.positions.calculators", None)

        module = importlib.import_module(
            "app.services.auto_strategy.positions.calculators"
        )

        assert "CalculatorFactory" not in module.__dict__

        exported = module.CalculatorFactory

        assert exported.__name__ == "CalculatorFactory"
        assert module.__dict__["CalculatorFactory"] is exported
        # 他の計算機はまだ読み込まれていない（個別遅延）
        assert "BaseCalculator" not in module.__dict__
        assert "HalfOptimalFCalculator" not in module.__dict__

    def test_optimization_package_is_removed(self) -> None:
        """optimization パッケージはチューニング機能削除に伴い削除済み。"""
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("app.services.auto_strategy.optimization")

    def test_generators_package_defers_random_gene_generator_import(self) -> None:
        """generators パッケージは属性アクセスまで個別モジュールを読み込まない。"""
        sys.modules.pop("app.services.auto_strategy.generators", None)

        module = importlib.import_module("app.services.auto_strategy.generators")

        assert "RandomGeneGenerator" not in module.__dict__

        exported = module.RandomGeneGenerator

        assert exported.__name__ == "RandomGeneGenerator"
        assert module.__dict__["RandomGeneGenerator"] is exported
        # 他のモジュールはまだ読み込まれていない（個別遅延）
        assert "ConditionGenerator" not in module.__dict__
        assert "SeedStrategyFactory" not in module.__dict__
