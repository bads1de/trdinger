"""
バックグラウンドタスク管理クラス

MLトレーニングやAutoML処理のバックグラウンドタスクのライフサイクルを管理します。
"""

import asyncio
import gc
import logging
import threading
import time
import weakref
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

logger = logging.getLogger(__name__)


class BackgroundTaskManager:
    """
    バックグラウンドタスクのライフサイクル管理クラス
    
    メモリリークを防ぎ、適切なリソース管理を行います。
    """

    def __init__(self):
        self._active_tasks: Dict[str, Dict[str, Any]] = {}
        self._task_resources: Dict[str, List[Any]] = {}
        self._cleanup_callbacks: Dict[str, List[callable]] = {}
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()

    def register_task(
        self, 
        task_id: Optional[str] = None,
        task_name: str = "Unknown Task",
        resources: Optional[List[Any]] = None,
        cleanup_callbacks: Optional[List[callable]] = None
    ) -> str:
        """
        バックグラウンドタスクを登録
        
        Args:
            task_id: タスクID（指定しない場合は自動生成）
            task_name: タスク名
            resources: 管理するリソースのリスト
            cleanup_callbacks: クリーンアップ時に呼び出すコールバック関数のリスト
            
        Returns:
            タスクID
        """
        if task_id is None:
            task_id = str(uuid4())
            
        with self._lock:
            self._active_tasks[task_id] = {
                "name": task_name,
                "start_time": time.time(),
                "status": "running",
                "memory_usage_start": self._get_memory_usage(),
            }
            
            self._task_resources[task_id] = resources or []
            self._cleanup_callbacks[task_id] = cleanup_callbacks or []
            
        logger.info(f"バックグラウンドタスク登録: {task_name} (ID: {task_id})")
        return task_id

    def unregister_task(self, task_id: str, force_cleanup: bool = True):
        """
        バックグラウンドタスクの登録を解除
        
        Args:
            task_id: タスクID
            force_cleanup: 強制的にクリーンアップを実行するか
        """
        with self._lock:
            if task_id not in self._active_tasks:
                logger.warning(f"未登録のタスクID: {task_id}")
                return
                
            task_info = self._active_tasks[task_id]
            task_name = task_info.get("name", "Unknown")
            
            if force_cleanup:
                self._cleanup_task_resources(task_id)
                
            # タスク情報を削除
            del self._active_tasks[task_id]
            self._task_resources.pop(task_id, None)
            self._cleanup_callbacks.pop(task_id, None)
            
            # メモリ使用量の変化を記録
            current_memory = self._get_memory_usage()
            start_memory = task_info.get("memory_usage_start", current_memory)
            memory_diff = current_memory - start_memory
            
            logger.info(
                f"バックグラウンドタスク終了: {task_name} (ID: {task_id}), "
                f"メモリ変化: {memory_diff:+.2f}MB"
            )

    def _cleanup_task_resources(self, task_id: str):
        """タスクのリソースをクリーンアップ"""
        try:
            # クリーンアップコールバックを実行
            callbacks = self._cleanup_callbacks.get(task_id, [])
            for callback in callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"クリーンアップコールバックエラー: {e}")
            
            # リソースをクリア
            resources = self._task_resources.get(task_id, [])
            for resource in resources:
                try:
                    if hasattr(resource, 'close'):
                        resource.close()
                    elif hasattr(resource, 'clear'):
                        resource.clear()
                    elif hasattr(resource, '__del__'):
                        del resource
                except Exception as e:
                    logger.error(f"リソースクリーンアップエラー: {e}")
            
            # 強制ガベージコレクション
            collected = gc.collect()
            logger.debug(f"タスク{task_id}のガベージコレクション: {collected}オブジェクト回収")
            
        except Exception as e:
            logger.error(f"タスクリソースクリーンアップエラー: {e}")

    @contextmanager
    def managed_task(
        self, 
        task_name: str,
        resources: Optional[List[Any]] = None,
        cleanup_callbacks: Optional[List[callable]] = None
    ):
        """
        管理されたバックグラウンドタスクのコンテキストマネージャー
        
        Args:
            task_name: タスク名
            resources: 管理するリソースのリスト
            cleanup_callbacks: クリーンアップ時に呼び出すコールバック関数のリスト
        """
        task_id = self.register_task(
            task_name=task_name,
            resources=resources,
            cleanup_callbacks=cleanup_callbacks
        )
        
        try:
            yield task_id
        finally:
            self.unregister_task(task_id, force_cleanup=True)

    def get_active_tasks(self) -> Dict[str, Dict[str, Any]]:
        """アクティブなタスクの一覧を取得"""
        with self._lock:
            return self._active_tasks.copy()

    def cleanup_all_tasks(self):
        """すべてのアクティブなタスクをクリーンアップ"""
        with self._lock:
            task_ids = list(self._active_tasks.keys())
            
        for task_id in task_ids:
            self.unregister_task(task_id, force_cleanup=True)
            
        logger.info(f"全バックグラウンドタスククリーンアップ完了: {len(task_ids)}個")

    def shutdown(self):
        """バックグラウンドタスクマネージャーをシャットダウン"""
        self._shutdown_event.set()
        self.cleanup_all_tasks()
        logger.info("バックグラウンドタスクマネージャーシャットダウン完了")

    def _get_memory_usage(self) -> float:
        """現在のメモリ使用量を取得（MB単位）"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
        except Exception as e:
            logger.warning(f"メモリ使用量取得エラー: {e}")
            return 0.0

    def get_memory_stats(self) -> Dict[str, Any]:
        """メモリ統計を取得"""
        current_memory = self._get_memory_usage()
        
        with self._lock:
            task_count = len(self._active_tasks)
            total_resources = sum(len(resources) for resources in self._task_resources.values())
            
        return {
            "current_memory_mb": current_memory,
            "active_task_count": task_count,
            "total_managed_resources": total_resources,
        }


# グローバルインスタンス
background_task_manager = BackgroundTaskManager()
