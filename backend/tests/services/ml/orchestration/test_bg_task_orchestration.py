"""
BackgroundTaskManager のユニットテスト

バックグラウンドタスクのライフサイクル管理をテストします:
- register_task / unregister_task
- managed_task (コンテキストマネージャー)
- cleanup_all_tasks
- リソース管理、コールバック実行
"""

import gc
from unittest.mock import MagicMock, patch

import pytest

from app.services.ml.orchestration.bg_task_orchestration_service import (
    BackgroundTaskManager,
)


@pytest.fixture
def manager() -> BackgroundTaskManager:
    return BackgroundTaskManager()


# ---------------------------------------------------------------------------
# register_task
# ---------------------------------------------------------------------------


class TestRegisterTask:
    def test_auto_generates_task_id(self, manager: BackgroundTaskManager):
        task_id = manager.register_task(task_name="test")
        assert task_id is not None
        assert isinstance(task_id, str)
        assert len(task_id) > 0

    def test_uses_provided_task_id(self, manager: BackgroundTaskManager):
        task_id = manager.register_task(task_id="my-id", task_name="test")
        assert task_id == "my-id"

    def test_registers_resources(self, manager: BackgroundTaskManager):
        resource = MagicMock()
        manager.register_task(task_id="r1", task_name="test", resources=[resource])
        assert resource in manager._task_resources["r1"]

    def test_registers_callbacks(self, manager: BackgroundTaskManager):
        cb = MagicMock()
        manager.register_task(task_id="c1", task_name="test", cleanup_callbacks=[cb])
        assert cb in manager._cleanup_callbacks["c1"]

    def test_default_task_name(self, manager: BackgroundTaskManager):
        tid = manager.register_task()
        assert manager._active_tasks[tid]["name"] == "Unknown Task"

    def test_task_status_is_running(self, manager: BackgroundTaskManager):
        tid = manager.register_task(task_name="t")
        assert manager._active_tasks[tid]["status"] == "running"


# ---------------------------------------------------------------------------
# unregister_task
# ---------------------------------------------------------------------------


class TestUnregisterTask:
    def test_removes_task(self, manager: BackgroundTaskManager):
        tid = manager.register_task(task_name="t")
        assert tid in manager._active_tasks

        manager.unregister_task(tid)
        assert tid not in manager._active_tasks

    def test_ignores_unknown_task_id(self, manager: BackgroundTaskManager):
        manager.unregister_task("nonexistent")  # should not raise

    def test_runs_cleanup_callbacks(self, manager: BackgroundTaskManager):
        cb = MagicMock()
        tid = manager.register_task(task_name="t", cleanup_callbacks=[cb])

        manager.unregister_task(tid)
        cb.assert_called_once()

    def test_closes_resources(self, manager: BackgroundTaskManager):
        resource = MagicMock()
        tid = manager.register_task(task_name="t", resources=[resource])

        manager.unregister_task(tid)
        resource.close.assert_called_once()

    def test_callback_error_does_not_break_unregister(
        self, manager: BackgroundTaskManager
    ):
        bad_cb = MagicMock(side_effect=Exception("boom"))
        good_cb = MagicMock()
        tid = manager.register_task(task_name="t", cleanup_callbacks=[bad_cb, good_cb])

        manager.unregister_task(tid)
        bad_cb.assert_called_once()
        good_cb.assert_called_once()
        assert tid not in manager._active_tasks

    def test_force_cleanup_false_skips_callbacks(self, manager: BackgroundTaskManager):
        cb = MagicMock()
        tid = manager.register_task(task_name="t", cleanup_callbacks=[cb])

        manager.unregister_task(tid, force_cleanup=False)
        cb.assert_not_called()
        assert tid not in manager._active_tasks


# ---------------------------------------------------------------------------
# managed_task
# ---------------------------------------------------------------------------


class TestManagedTask:
    def test_yields_task_id(self, manager: BackgroundTaskManager):
        with manager.managed_task("test_task") as tid:
            assert tid in manager._active_tasks

    def test_cleans_up_on_exit(self, manager: BackgroundTaskManager):
        tid_holder = []

        with manager.managed_task("test_task") as tid:
            tid_holder.append(tid)
            assert tid in manager._active_tasks

        assert tid_holder[0] not in manager._active_tasks

    def test_cleans_up_on_exception(self, manager: BackgroundTaskManager):
        tid_holder = []

        with pytest.raises(RuntimeError):
            with manager.managed_task("failing_task") as tid:
                tid_holder.append(tid)
                raise RuntimeError("task failed")

        assert tid_holder[0] not in manager._active_tasks

    def test_callbacks_called_on_exit(self, manager: BackgroundTaskManager):
        cb = MagicMock()

        with manager.managed_task("t", cleanup_callbacks=[cb]):
            pass

        cb.assert_called_once()

    def test_managed_task_with_resources(self, manager: BackgroundTaskManager):
        resource = MagicMock()

        with manager.managed_task("t", resources=[resource]):
            pass

        resource.close.assert_called_once()


# ---------------------------------------------------------------------------
# cleanup_all_tasks
# ---------------------------------------------------------------------------


class TestCleanupAllTasks:
    def test_clears_all_tasks(self, manager: BackgroundTaskManager):
        manager.register_task(task_id="t1", task_name="task1")
        manager.register_task(task_id="t2", task_name="task2")
        assert len(manager._active_tasks) == 2

        manager.cleanup_all_tasks()
        assert len(manager._active_tasks) == 0

    def test_empty_manager_does_nothing(self, manager: BackgroundTaskManager):
        manager.cleanup_all_tasks()  # should not raise
        assert len(manager._active_tasks) == 0

    def test_callbacks_called_for_all(self, manager: BackgroundTaskManager):
        cb1 = MagicMock()
        cb2 = MagicMock()
        manager.register_task(task_id="t1", task_name="t1", cleanup_callbacks=[cb1])
        manager.register_task(task_id="t2", task_name="t2", cleanup_callbacks=[cb2])

        manager.cleanup_all_tasks()

        cb1.assert_called_once()
        cb2.assert_called_once()


# ---------------------------------------------------------------------------
# _cleanup_task_resources
# ---------------------------------------------------------------------------


class TestCleanupTaskResources:
    def test_calls_clear_on_resource(self, manager: BackgroundTaskManager):
        resource = MagicMock(spec=["clear"])
        tid = manager.register_task(task_name="t", resources=[resource])

        manager._cleanup_task_resources(tid)
        resource.clear.assert_called_once()

    def test_handles_resource_error(self, manager: BackgroundTaskManager):
        resource = MagicMock()
        resource.close.side_effect = Exception("close failed")
        tid = manager.register_task(task_name="t", resources=[resource])

        manager._cleanup_task_resources(tid)  # should not raise


# ---------------------------------------------------------------------------
# グローバルインスタンス
# ---------------------------------------------------------------------------


class TestGlobalInstance:
    def test_background_task_manager_exists(self):
        from app.services.ml.orchestration.bg_task_orchestration_service import (
            background_task_manager,
        )

        assert isinstance(background_task_manager, BackgroundTaskManager)
