"""
Auto Strategy Services モジュール

自動戦略生成に関連するサービスクラスを提供します。
managers/, persistence/ の機能を統合しています。
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .auto_strategy_service import AutoStrategyService
    from .experiment_application_service import ExperimentApplicationService
    from .experiment_backtest_service import ExperimentBacktestService
    from .experiment_engine_registry import ExperimentEngineRegistry
    from .experiment_manager import ExperimentManager
    from .experiment_persistence_service import ExperimentPersistenceService
    from ..positions.position_sizing_service import PositionSizingService
    from ..tpsl.tpsl_service import TPSLService

_ATTRIBUTE_EXPORTS = {
    "AutoStrategyService": ".auto_strategy_service",
    "ExperimentApplicationService": ".experiment_application_service",
    "ExperimentBacktestService": ".experiment_backtest_service",
    "ExperimentEngineRegistry": ".experiment_engine_registry",
    "ExperimentManager": ".experiment_manager",
    "ExperimentPersistenceService": ".experiment_persistence_service",
    "PositionSizingService": "..positions.position_sizing_service",
    "TPSLService": "..tpsl.tpsl_service",
}


def __getattr__(name: str) -> Any:
    module_path = _ATTRIBUTE_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *_ATTRIBUTE_EXPORTS})


__all__ = [
    "AutoStrategyService",
    "PositionSizingService",
    "TPSLService",
    "ExperimentBacktestService",
    "ExperimentApplicationService",
    "ExperimentEngineRegistry",
    "ExperimentManager",
    "ExperimentPersistenceService",
]
