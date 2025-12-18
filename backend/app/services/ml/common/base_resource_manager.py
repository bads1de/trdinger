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
        """リソースの統一クリーンアップ"""
        if self._is_cleaned_up:
            return {"status": "already_cleaned", "memory_freed": 0}

        lvl = level or self._cleanup_level
        mem_before = self._get_memory_usage()
        stats = {"level": lvl.value, "memory_before": mem_before, "errors": [], "cleaned": []}

        # クリーンアップフェーズの定義
        phases = [
            ("_cleanup_temporary_files", "temporary_files"),
            ("_cleanup_cache", "cache"),
            ("_cleanup_models", "models"),
            ("_cleanup_other_resources", "other_resources")
        ]

        for method_name, name in phases:
            try:
                getattr(self, method_name)(lvl)
                stats["cleaned"].append(name)
            except Exception as e:
                err = f"{name} cleanup error: {e}"
                logger.warning(err)
                stats["errors"].append(err)

        if lvl in [CleanupLevel.STANDARD, CleanupLevel.THOROUGH]:
            stats["objects_collected"] = self._force_garbage_collection()

        mem_after = self._get_memory_usage()
        stats.update({"memory_after": mem_after, "memory_freed": mem_before - mem_after})
        self._is_cleaned_up = True

        logger.info(f"Cleanup done: {stats['memory_freed']:.2f}MB freed")
        return stats

    @abstractmethod
    def _cleanup_temporary_files(self, level: CleanupLevel):
        """一時ファイルのクリーンアップ（サブクラスで実装）"""

    @abstractmethod
    def _cleanup_cache(self, level: CleanupLevel):
        """キャッシュのクリーンアップ（サブクラスで実装）"""

    @abstractmethod
    def _cleanup_models(self, level: CleanupLevel):
        """モデルオブジェクトのクリーンアップ（サブクラスで実装）"""

    def _cleanup_other_resources(self, level: CleanupLevel):
        """その他のリソースクリーンアップ（オプション、サブクラスでオーバーライド可能）"""

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



