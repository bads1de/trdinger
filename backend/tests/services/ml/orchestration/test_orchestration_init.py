"""
orchestrationパッケージの__init__.pyのテスト

遅延ロード機能（__getattr__）とエクスポート定義を確認します。
"""

import pytest

import app.services.ml.orchestration as orchestration_package


class TestMLOrchestrationInitExports:
    """orchestration/__init__.pyのエクスポートテスト"""

    def test_background_task_manager_lazy_load(self):
        """BackgroundTaskManagerが遅延ロードされる"""
        from app.services.ml.orchestration.bg_task_orchestration_service import (
            BackgroundTaskManager,
        )

        manager = getattr(orchestration_package, "BackgroundTaskManager")

        assert manager is BackgroundTaskManager

    def test_background_task_manager_instance_lazy_load(self):
        """background_task_managerインスタンスが遅延ロードされる"""
        from app.services.ml.orchestration.bg_task_orchestration_service import (
            background_task_manager,
        )

        manager = getattr(orchestration_package, "background_task_manager")

        assert manager is background_task_manager

    def test_ml_management_orchestration_service_lazy_load(self):
        """MLManagementOrchestrationServiceが遅延ロードされる"""
        from app.services.ml.orchestration.ml_management_orchestration_service import (
            MLManagementOrchestrationService,
        )

        service = getattr(orchestration_package, "MLManagementOrchestrationService")

        assert service is MLManagementOrchestrationService

    def test_ml_training_service_lazy_load(self):
        """MLTrainingServiceが遅延ロードされる"""
        from app.services.ml.orchestration.ml_training_orchestration_service import (
            MLTrainingService,
        )

        service = getattr(orchestration_package, "MLTrainingService")

        assert service is MLTrainingService

    def test_ml_training_service_instance_lazy_load(self):
        """ml_training_serviceインスタンスが遅延ロードされる"""
        from app.services.ml.orchestration.ml_training_orchestration_service import (
            ml_training_service,
        )

        service = getattr(orchestration_package, "ml_training_service")

        assert service is ml_training_service

    def test_getattr_raises_for_non_existent(self):
        """存在しない属性でAttributeErrorが発生する"""
        with pytest.raises(AttributeError, match="module.*has no attribute"):
            _ = orchestration_package.NonExistentAttribute

    def test_all_contains_expected_items(self):
        """__all__に期待されるアイテムが含まれる"""
        expected_items = [
            "MLManagementOrchestrationService",
            "MLTrainingService",
            "ml_training_service",
            "BackgroundTaskManager",
            "background_task_manager",
        ]

        for item in expected_items:
            assert item in orchestration_package.__all__, f"{item} not in __all__"

    def test_all_is_list(self):
        """__all__がリストである"""
        assert isinstance(orchestration_package.__all__, list)

    def test_module_has_docstring(self):
        """モジュールにドキュメント文字列がある"""
        assert orchestration_package.__doc__ is not None
        assert len(orchestration_package.__doc__) > 0
