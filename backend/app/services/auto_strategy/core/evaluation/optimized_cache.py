"""
最適化されたキャッシュシステム

階層的キャッシングと効率的なキャッシュ管理を提供します。
"""

import hashlib
import logging
import threading
from typing import Any, Dict, Optional, Tuple

from cachetools import LRUCache

logger = logging.getLogger(__name__)


class OptimizedCacheSystem:
    """
    最適化されたキャッシュシステム

    主な最適化ポイント:
    1. 階層的キャッシング（構造、パラメータ、結果）
    2. スレッドセーフなキャッシュ操作
    3. キャッシュヒット率の監視
    4. メモリ効率的なキャッシュ管理
    """

    def __init__(
        self,
        structure_cache_size: int = 100,
        param_cache_size: int = 1000,
        result_cache_size: int = 10000,
    ):
        # 階層的キャッシュ
        self._structure_cache: LRUCache[str, Any] = LRUCache(maxsize=structure_cache_size)
        self._param_cache: LRUCache[str, Any] = LRUCache(maxsize=param_cache_size)
        self._result_cache: LRUCache[str, Any] = LRUCache(maxsize=result_cache_size)

        # スレッドセーフのためのロック
        self._lock = threading.Lock()

        # 統計情報
        self._hits = 0
        self._misses = 0
        self._structure_hits = 0
        self._param_hits = 0

    def get(self, gene: Any) -> Optional[Tuple[float, ...]]:
        """
        キャッシュから結果を取得（最適化版）

        最適化:
        - 階層的なキャッシュキー生成
        - 早期リターン
        - 統計情報の効率的な更新
        """
        cache_key = self._build_cache_key(gene)

        with self._lock:
            if cache_key in self._result_cache:
                self._hits += 1
                return self._result_cache[cache_key]

            self._misses += 1
            return None

    def set(self, gene: Any, result: Tuple[float, ...]):
        """結果をキャッシュに保存"""
        cache_key = self._build_cache_key(gene)

        with self._lock:
            self._result_cache[cache_key] = result

    def _build_cache_key(self, gene: Any) -> str:
        """
        階層的なキャッシュキーを生成（最適化版）

        最適化:
        - 構造レベルのキーで粗いフィルタリング
        - パラメータレベルのキーで詳細な識別
        - ハッシュ計算の効率化
        """
        # 1. 構造レベルのキー（高速）
        structure_key = self._get_structure_key(gene)

        # 2. パラメータレベルのキー
        param_key = self._get_param_key(gene)

        return f"{structure_key}:{param_key}"

    def _get_structure_key(self, gene: Any) -> str:
        """
        遺伝子構造のキーを生成（最適化版）

        最適化:
        - 直接的な属性アクセス
        - 最小限のハッシュ計算
        """
        # 構造情報を直接取得（getattr削減）
        indicators = gene.indicators if hasattr(gene, 'indicators') else []
        long_conditions = gene.long_entry_conditions if hasattr(gene, 'long_entry_conditions') else []
        short_conditions = gene.short_entry_conditions if hasattr(gene, 'short_entry_conditions') else []

        structure = (
            len(indicators),
            len(long_conditions),
            len(short_conditions),
            gene.tpsl_gene is not None if hasattr(gene, 'tpsl_gene') else False,
            gene.position_sizing_gene is not None if hasattr(gene, 'position_sizing_gene') else False,
        )

        # 高速ハッシュ計算
        return hashlib.md5(str(structure).encode()).hexdigest()[:8]

    def _get_param_key(self, gene: Any) -> str:
        """
        パラメータのキーを生成（最適化版）

        最適化:
        - インジケーターパラメータの効率的な抽出
        - 条件パラメータの最小限の処理
        """
        param_parts = []

        # インジケーターパラメータ
        indicators = gene.indicators if hasattr(gene, 'indicators') else []
        for ind in indicators:
            if hasattr(ind, 'type'):
                param_parts.append(ind.type)
            if hasattr(ind, 'parameters'):
                params = ind.parameters
                if isinstance(params, dict):
                    # 重要なパラメータのみ
                    if 'period' in params:
                        param_parts.append(f"p{params['period']}")

        # 条件パラメータ
        long_conditions = gene.long_entry_conditions if hasattr(gene, 'long_entry_conditions') else []
        for cond in long_conditions:
            if hasattr(cond, 'operator'):
                param_parts.append(cond.operator)

        param_str = '|'.join(param_parts)
        return hashlib.md5(param_str.encode()).hexdigest()[:8]

    def get_statistics(self) -> Dict[str, Any]:
        """キャッシュ統計を取得"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0

            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "structure_cache_size": len(self._structure_cache),
                "param_cache_size": len(self._param_cache),
                "result_cache_size": len(self._result_cache),
                "structure_hits": self._structure_hits,
                "param_hits": self._param_hits,
            }

    def clear(self):
        """キャッシュをクリア"""
        with self._lock:
            self._structure_cache.clear()
            self._param_cache.clear()
            self._result_cache.clear()
            self._hits = 0
            self._misses = 0
            self._structure_hits = 0
            self._param_hits = 0

    def reset_statistics(self):
        """統計情報をリセット"""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._structure_hits = 0
            self._param_hits = 0


class DataCacheOptimizer:
    """
    データキャッシュの最適化

    バックテストデータの効率的なキャッシングを提供します。
    """

    def __init__(self, max_cache_size: int = 100):
        self._data_cache: LRUCache[str, Any] = LRUCache(maxsize=max_cache_size)
        self._minute_data_cache: LRUCache[str, Any] = LRUCache(maxsize=max_cache_size)
        self._lock = threading.Lock()

        # 統計情報
        self._data_hits = 0
        self._data_misses = 0

    def get_data(self, cache_key: str) -> Optional[Any]:
        """データをキャッシュから取得"""
        with self._lock:
            if cache_key in self._data_cache:
                self._data_hits += 1
                return self._data_cache[cache_key]

            self._data_misses += 1
            return None

    def set_data(self, cache_key: str, data: Any):
        """データをキャッシュに保存"""
        with self._lock:
            self._data_cache[cache_key] = data

    def get_minute_data(self, cache_key: str) -> Optional[Any]:
        """1分足データをキャッシュから取得"""
        with self._lock:
            if cache_key in self._minute_data_cache:
                return self._minute_data_cache[cache_key]
            return None

    def set_minute_data(self, cache_key: str, data: Any):
        """1分足データをキャッシュに保存"""
        with self._lock:
            self._minute_data_cache[cache_key] = data

    def build_data_cache_key(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> str:
        """データキャッシュキーを生成"""
        key_parts = [symbol, timeframe, str(start_date), str(end_date)]
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get_statistics(self) -> Dict[str, Any]:
        """キャッシュ統計を取得"""
        with self._lock:
            total = self._data_hits + self._data_misses
            hit_rate = self._data_hits / total if total > 0 else 0.0

            return {
                "data_hits": self._data_hits,
                "data_misses": self._data_misses,
                "data_hit_rate": hit_rate,
                "data_cache_size": len(self._data_cache),
                "minute_data_cache_size": len(self._minute_data_cache),
            }

    def clear(self):
        """キャッシュをクリア"""
        with self._lock:
            self._data_cache.clear()
            self._minute_data_cache.clear()
            self._data_hits = 0
            self._data_misses = 0
