"""
Auto Strategy Services モジュール

自動戦略生成に関連するサービスクラスを提供します。
managers/, persistence/ の機能を統合しています。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..positions.position_sizing_service import PositionSizingService
    from ..tpsl.tpsl_service import TPSLService
    from .auto_strategy_service import AutoStrategyService
    from .experiment_application_service import ExperimentApplicationService
    from .experiment_backtest_service import ExperimentBacktestService
    from .experiment_engine_registry import ExperimentEngineRegistry
    from .experiment_manager import ExperimentManager
    from .experiment_persistence_service import ExperimentPersistenceService

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

from .._lazy_import import setup_lazy_import  # noqa: E402

setup_lazy_import(globals(), _ATTRIBUTE_EXPORTS, __all__)
