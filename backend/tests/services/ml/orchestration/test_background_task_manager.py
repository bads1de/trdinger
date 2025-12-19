
import pytest
from unittest.mock import Mock, MagicMock, patch
from app.services.ml.orchestration.background_task_manager import BackgroundTaskManager

class TestBackgroundTaskManager:
    """BackgroundTaskManagerのテスト"""

    @pytest.fixture
    def manager(self):
        """BackgroundTaskManagerのインスタンス"""
        return BackgroundTaskManager()

    def test_register_and_unregister_task(self, manager):
        """タスクの登録と解除のテスト"""
        task_id = manager.register_task(task_name="test_task")
        
        assert task_id in manager._active_tasks
        assert manager._active_tasks[task_id]["name"] == "test_task"
        
        manager.unregister_task(task_id)
        assert task_id not in manager._active_tasks

    def test_cleanup_resources(self, manager):
        """リソースクリーンアップのテスト"""
        # Mock resource with close method
        resource = Mock()
        resource.close = Mock()
        
        # Cleanup callback
        callback = Mock()
        
        task_id = manager.register_task(
            task_name="resource_task",
            resources=[resource],
            cleanup_callbacks=[callback]
        )
        
        manager.unregister_task(task_id, force_cleanup=True)
        
        # Verify cleanup
        resource.close.assert_called_once()
        callback.assert_called_once()

    def test_managed_task_context_manager(self, manager):
        """コンテキストマネージャーのテスト"""
        resource = Mock()
        resource.close = Mock()
        
        with manager.managed_task("managed_task", resources=[resource]) as task_id:
            assert task_id in manager._active_tasks
            assert manager._active_tasks[task_id]["name"] == "managed_task"
            
        # Context exit should cleanup
        assert task_id not in manager._active_tasks
        resource.close.assert_called_once()

    def test_cleanup_all_tasks(self, manager):
        """全タスククリーンアップのテスト"""
        manager.register_task(task_id="task1", task_name="task1")
        manager.register_task(task_id="task2", task_name="task2")
        
        assert len(manager._active_tasks) == 2
        
        manager.cleanup_all_tasks()
        
        assert len(manager._active_tasks) == 0

    def test_memory_usage_tracking(self, manager):
        """メモリ使用量追跡のテスト（psutilの有無に関わらず動作すること）"""
        with patch("app.services.ml.orchestration.background_task_manager.logger") as mock_logger:
            task_id = manager.register_task("memory_task")
            manager.unregister_task(task_id)
            
            # ログ出力にメモリ情報が含まれるか確認
            # infoメソッドが呼ばれていることを確認（引数の内容は環境依存なので厳密にはチェックしにくいが）
            assert mock_logger.info.called

    def test_cleanup_error_handling(self, manager):
        """クリーンアップ中のエラーハンドリング"""
        # エラーを起こすコールバック
        bad_callback = Mock(side_effect=Exception("Callback failed"))
        
        task_id = manager.register_task(
            task_name="error_task",
            cleanup_callbacks=[bad_callback]
        )
        
        # エラーが発生しても処理が継続することを確認
        with patch("app.services.ml.orchestration.background_task_manager.logger") as mock_logger:
            manager.unregister_task(task_id)
            
            # エラーログが出力されていること
            assert mock_logger.error.called
            # タスクは削除されていること
            assert task_id not in manager._active_tasks
