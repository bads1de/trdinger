"""
ML Orchestration パッケージ

MLトレーニングの管理と自動化機能を提供します。
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bg_task_orchestration_service import (
        BackgroundTaskManager,
        background_task_manager,
    )
    from .ml_management_orchestration_service import MLManagementOrchestrationService
    from .ml_training_orchestration_service import (
        MLTrainingService,
        ml_training_service,
    )

_ATTRIBUTE_EXPORTS = {
    "BackgroundTaskManager": ".bg_task_orchestration_service",
    "background_task_manager": ".bg_task_orchestration_service",
    "MLManagementOrchestrationService": ".ml_management_orchestration_service",
    "MLTrainingService": ".ml_training_orchestration_service",
    "ml_training_service": ".ml_training_orchestration_service",
}


def __getattr__(name: str) -> type:
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
    # Core services
    "MLManagementOrchestrationService",
    "MLTrainingService",
    "ml_training_service",
    # Background task management
    "BackgroundTaskManager",
    "background_task_manager",
]
