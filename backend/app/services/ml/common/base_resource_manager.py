"""
BaseResourceManager - 統一されたリソース管理インターフェース

このモジュールは、ML関連クラスのリソースクリーンアップロジックの重複を解消するため、
統一されたリソース管理インターフェースを提供します。
"""

import gc
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CleanupLevel(Enum):
    """クリーンアップレベルの定義"""

    MINIMAL = "minimal"  # 基本的なクリーンアップのみ
    STANDARD = "standard"  # 通常のクリーンアップ
    THOROUGH = "thorough"  # 徹底的なクリーンアップ（メモリ最適化重視）


class BaseResourceManager(ABC):
    """
    リソース管理の抽象基底クラス

    統一されたリソースクリーンアップインターフェースを提供し、
    メモリリークの防止と保守性の向上を実現します。
    """

    def __init__(self):
        self._cleanup_level = CleanupLevel.STANDARD
        self._cleanup_callbacks: List[callable] = []
        self._is_cleaned_up = False

    def __enter__(self):
        """コンテキストマネージャーの開始"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了時に自動クリーンアップ"""
        try:
            self.cleanup_resources()
        except Exception as e:
            logger.error(f"コンテキストマネージャー終了時のクリーンアップエラー: {e}")

    def set_cleanup_level(self, level: CleanupLevel):
        """クリーンアップレベルを設定"""
        self._cleanup_level = level

    def add_cleanup_callback(self, callback: callable):
        """クリーンアップ時に実行するコールバックを追加"""
        if callable(callback):
            self._cleanup_callbacks.append(callback)

    def cleanup_resources(self, level: Optional[CleanupLevel] = None) -> Dict[str, Any]:
        """
        リソースの統一クリーンアップ

        Args:
            level: クリーンアップレベル（指定しない場合はデフォルトレベルを使用）

        Returns:
            クリーンアップ結果の統計情報
        """
        if self._is_cleaned_up:
            logger.debug("リソースは既にクリーンアップ済みです")
            return {"status": "already_cleaned", "memory_freed": 0}

        cleanup_level = level or self._cleanup_level
        memory_before = self._get_memory_usage()
        cleanup_stats = {
            "level": cleanup_level.value,
            "memory_before": memory_before,
            "errors": [],
            "cleaned_components": [],
        }

        try:
            logger.info(
                f"リソースクリーンアップを開始（レベル: {cleanup_level.value}）"
            )

            # 1. 一時ファイルのクリーンアップ
            try:
                self._cleanup_temporary_files(cleanup_level)
                cleanup_stats["cleaned_components"].append("temporary_files")
            except Exception as e:
                error_msg = f"一時ファイルクリーンアップエラー: {e}"
                logger.warning(error_msg)
                cleanup_stats["errors"].append(error_msg)

            # 2. キャッシュのクリーンアップ
            try:
                self._cleanup_cache(cleanup_level)
                cleanup_stats["cleaned_components"].append("cache")
            except Exception as e:
                error_msg = f"キャッシュクリーンアップエラー: {e}"
                logger.warning(error_msg)
                cleanup_stats["errors"].append(error_msg)

            # 3. モデルオブジェクトのクリーンアップ
            try:
                self._cleanup_models(cleanup_level)
                cleanup_stats["cleaned_components"].append("models")
            except Exception as e:
                error_msg = f"モデルクリーンアップエラー: {e}"
                logger.warning(error_msg)
                cleanup_stats["errors"].append(error_msg)

            # 4. その他のリソースクリーンアップ
            try:
                self._cleanup_other_resources(cleanup_level)
                cleanup_stats["cleaned_components"].append("other_resources")
            except Exception as e:
                error_msg = f"その他リソースクリーンアップエラー: {e}"
                logger.warning(error_msg)
                cleanup_stats["errors"].append(error_msg)

            # 5. コールバック実行
            self._execute_cleanup_callbacks(cleanup_stats)

            # 6. ガベージコレクション
            if cleanup_level in [CleanupLevel.STANDARD, CleanupLevel.THOROUGH]:
                collected = self._force_garbage_collection()
                cleanup_stats["objects_collected"] = collected

            # 7. メモリ使用量の計算
            memory_after = self._get_memory_usage()
            cleanup_stats["memory_after"] = memory_after
            cleanup_stats["memory_freed"] = memory_before - memory_after

            self._is_cleaned_up = True

            logger.info(
                f"リソースクリーンアップ完了: "
                f"{cleanup_stats['memory_freed']:.2f}MB解放, "
                f"{len(cleanup_stats['cleaned_components'])}コンポーネント処理"
            )

            return cleanup_stats

        except Exception as e:
            error_msg = f"リソースクリーンアップ中に予期しないエラー: {e}"
            logger.error(error_msg)
            cleanup_stats["errors"].append(error_msg)
            cleanup_stats["status"] = "failed"
            return cleanup_stats

    @abstractmethod
    def _cleanup_temporary_files(self, level: CleanupLevel):
        """一時ファイルのクリーンアップ（サブクラスで実装）"""
        pass

    @abstractmethod
    def _cleanup_cache(self, level: CleanupLevel):
        """キャッシュのクリーンアップ（サブクラスで実装）"""
        pass

    @abstractmethod
    def _cleanup_models(self, level: CleanupLevel):
        """モデルオブジェクトのクリーンアップ（サブクラスで実装）"""
        pass

    def _cleanup_other_resources(self, level: CleanupLevel):
        """その他のリソースクリーンアップ（オプション、サブクラスでオーバーライド可能）"""
        pass

    def _execute_cleanup_callbacks(self, cleanup_stats: Dict[str, Any]):
        """登録されたクリーンアップコールバックを実行"""
        for i, callback in enumerate(self._cleanup_callbacks):
            try:
                callback()
                logger.debug(f"クリーンアップコールバック{i}実行完了")
            except Exception as e:
                error_msg = f"クリーンアップコールバック{i}実行エラー: {e}"
                logger.warning(error_msg)
                cleanup_stats["errors"].append(error_msg)

    def _force_garbage_collection(self) -> int:
        """強制ガベージコレクション"""
        try:
            collected = gc.collect()
            logger.debug(f"ガベージコレクション実行: {collected}オブジェクト回収")
            return collected
        except Exception as e:
            logger.error(f"ガベージコレクションエラー: {e}")
            return 0

    def _get_memory_usage(self) -> float:
        """現在のメモリ使用量を取得（MB単位）"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            logger.debug("psutilが利用できません。メモリ使用量の取得をスキップします")
            return 0.0
        except Exception as e:
            logger.warning(f"メモリ使用量取得エラー: {e}")
            return 0.0

    def is_cleaned_up(self) -> bool:
        """クリーンアップ済みかどうかを確認"""
        return self._is_cleaned_up


class ResourceManagedOperation:
    """
    リソース管理付きの操作を実行するコンテキストマネージャー

    with文を使用して、操作の開始時と終了時に自動的にリソース管理を行います。
    """

    def __init__(
        self,
        resource_manager: BaseResourceManager,
        operation_name: str = "operation",
        cleanup_level: CleanupLevel = CleanupLevel.STANDARD,
    ):
        self.resource_manager = resource_manager
        self.operation_name = operation_name
        self.cleanup_level = cleanup_level
        self.start_memory = 0.0
        self.start_time = None

    def __enter__(self):
        """操作開始時の処理"""
        import time

        self.start_time = time.time()
        self.start_memory = self.resource_manager._get_memory_usage()
        logger.info(
            f"{self.operation_name}開始（メモリ使用量: {self.start_memory:.2f}MB）"
        )
        return self.resource_manager

    def __exit__(self, exc_type, exc_val, exc_tb):
        """操作終了時の自動クリーンアップ"""
        import time

        try:
            # クリーンアップ実行
            cleanup_stats = self.resource_manager.cleanup_resources(self.cleanup_level)

            # 統計情報の計算
            end_time = time.time()
            duration = end_time - self.start_time if self.start_time else 0
            end_memory = self.resource_manager._get_memory_usage()
            memory_change = end_memory - self.start_memory

            logger.info(
                f"{self.operation_name}完了: "
                f"実行時間={duration:.2f}秒, "
                f"メモリ変化={memory_change:+.2f}MB, "
                f"解放={cleanup_stats.get('memory_freed', 0):.2f}MB"
            )

        except Exception as e:
            logger.error(f"{self.operation_name}終了時のクリーンアップエラー: {e}")


@contextmanager
def managed_ml_operation(
    resource_manager: BaseResourceManager,
    operation_name: str = "ML Operation",
    cleanup_level: CleanupLevel = CleanupLevel.STANDARD,
):
    """
    ML操作用のコンテキストマネージャー（関数形式）

    使用例:
        with managed_ml_operation(trainer, "モデル学習") as managed_trainer:
            result = managed_trainer.train_model(data)
    """
    with ResourceManagedOperation(
        resource_manager, operation_name, cleanup_level
    ) as manager:
        yield manager
