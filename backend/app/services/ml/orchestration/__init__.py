"""
ML Orchestration パッケージ

MLトレーニングの管理と自動化機能を提供します。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bg_task_orchestration_service import (
        BackgroundTaskManager,
        background_task_manager,
    )
    from .ml_management_orchestration_service import (
        MLManagementOrchestrationService,
    )
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

__all__ = [
    # Core services
    "MLManagementOrchestrationService",
    "MLTrainingService",
    "ml_training_service",
    # Background task management
    "BackgroundTaskManager",
    "background_task_manager",
]

from app.utils.lazy_import import setup_lazy_import  # noqa: E402

setup_lazy_import(globals(), _ATTRIBUTE_EXPORTS)
