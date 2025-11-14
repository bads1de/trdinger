"""
BaseResourceManager - 統一されたリソース管理インターフェース

このモジュールは、ML関連クラスのリソースクリーンアップロジックの重複を解消するため、
統一されたリソース管理インターフェースを提供します。
"""

import gc
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CleanupLevel(Enum):
    """クリーンアップレベルの定義"""

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
        self._is_cleaned_up = False

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

            # 5. ガベージコレクション
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

    def _force_garbage_collection(self) -> int:
        """強制ガベージコレクション"""
        try:
            collected = gc.collect()
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
            return 0.0
        except Exception as e:
            logger.warning(f"メモリ使用量取得エラー: {e}")
            return 0.0
