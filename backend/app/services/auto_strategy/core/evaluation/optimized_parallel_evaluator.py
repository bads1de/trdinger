"""
最適化された並列評価モジュール

パフォーマンス最適化版の並列評価を提供します。
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OptimizedParallelEvaluator:
    """
    最適化された並列評価クラス

    主な最適化ポイント:
    1. スレッドプールの効率的な活用（プロセスプールのオーバーヘッド削減）
    2. バッチ処理によるタスクスケジューリング最適化
    3. 動的ワーカー調整
    4. キャッシュとの統合
    """

    def __init__(
        self,
        evaluate_func: Callable[[Any], Tuple[float, ...]],
        max_workers: Optional[int] = None,
        timeout_per_individual: float = 300.0,
        batch_size: int = 10,
        use_cache: bool = True,
    ):
        """
        初期化

        Args:
            evaluate_func: 個体を評価する関数
            max_workers: 最大ワーカー数（Noneの場合はCPUコア数）
            timeout_per_individual: 個体あたりのタイムアウト秒数
            batch_size: バッチサイズ
            use_cache: キャッシュを使用するか
        """
        self.evaluate_func = evaluate_func
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) * 2)
        self.timeout_per_individual = timeout_per_individual
        self.batch_size = batch_size
        self.use_cache = use_cache

        # キャッシュ
        self._cache: Dict[str, Tuple[float, ...]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # スレッドプール
        self._executor: Optional[ThreadPoolExecutor] = None

        # 統計情報
        self._total_evaluations = 0
        self._successful_evaluations = 0
        self._failed_evaluations = 0
        self._timeout_evaluations = 0

    def start(self):
        """Executorを起動"""
        if self._executor is not None:
            return

        logger.info(f"最適化並列評価Executorを起動 (max_workers={self.max_workers})")
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def shutdown(self):
        """Executorを停止"""
        if self._executor:
            logger.info("最適化並列評価Executorを停止")
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def evaluate_population(
        self,
        population: List[Any],
        default_fitness: Optional[Tuple[float, ...]] = None,
    ) -> List[Tuple[float, ...]]:
        """
        個体群を並列評価（最適化版）

        最適化:
        - バッチ処理によるオーバーヘッド削減
        - キャッシュの活用
        - 動的タスクスケジューリング
        """
        if not population:
            return []

        if default_fitness is None:
            default_fitness = (0.0,)

        # Executorが起動していない場合は起動
        if self._executor is None:
            self.start()

        results: List[Tuple[float, ...]] = []
        population_size = len(population)

        logger.info(
            f"最適化並列評価開始: {population_size}個体, "
            f"{self.max_workers}ワーカー, バッチサイズ={self.batch_size}"
        )

        start_time = time.perf_counter()

        # バッチ処理
        for batch_start in range(0, population_size, self.batch_size):
            batch_end = min(batch_start + self.batch_size, population_size)
            batch = population[batch_start:batch_end]

            batch_results = self._evaluate_batch(batch, default_fitness)
            results.extend(batch_results)

        elapsed = time.perf_counter() - start_time

        logger.info(
            f"最適化並列評価完了: {len(results)}個体, "
            f"成功={self._successful_evaluations}, "
            f"失敗={self._failed_evaluations}, "
            f"キャッシュヒット={self._cache_hits}, "
            f"時間={elapsed:.2f}秒"
        )

        return results

    def _evaluate_batch(
        self,
        batch: List[Any],
        default_fitness: Tuple[float, ...],
    ) -> List[Tuple[float, ...]]:
        """バッチ評価"""
        batch_size = len(batch)
        results: List[Optional[Tuple[float, ...]]] = [None] * batch_size
        future_to_index = {}

        # キャッシュチェックとタスク提出
        tasks_to_submit = []
        for i, ind in enumerate(batch):
            cache_key = self._get_cache_key(ind)

            if self.use_cache and cache_key in self._cache:
                results[i] = self._cache[cache_key]
                self._cache_hits += 1
            else:
                tasks_to_submit.append((i, ind, cache_key))

        # タスクを提出
        for i, ind, cache_key in tasks_to_submit:
            future = self._executor.submit(self._evaluate_with_cache, ind, cache_key)  # type: ignore[union-attr]
            future_to_index[future] = i

        # 結果を収集
        batch_timeout = self.timeout_per_individual * len(tasks_to_submit)

        try:
            for future in as_completed(future_to_index, timeout=batch_timeout):
                index = future_to_index[future]
                self._total_evaluations += 1

                try:
                    fitness = future.result(timeout=self.timeout_per_individual)
                    results[index] = fitness
                    self._successful_evaluations += 1
                except TimeoutError:
                    logger.warning(f"個体評価タイムアウト: index={index}")
                    results[index] = default_fitness
                    self._timeout_evaluations += 1
                except Exception as e:
                    logger.warning(f"個体評価エラー: index={index}, error={e}")
                    results[index] = default_fitness
                    self._failed_evaluations += 1
        except TimeoutError:
            logger.warning("バッチ全体タイムアウト")
            for i, r in enumerate(results):
                if r is None:
                    results[i] = default_fitness
                    self._timeout_evaluations += 1

        # Noneをデフォルト値で埋める
        return [r if r is not None else default_fitness for r in results]

    def _evaluate_with_cache(
        self,
        individual: Any,
        cache_key: str,
    ) -> Tuple[float, ...]:
        """キャッシュ付き評価"""
        try:
            fitness = self.evaluate_func(individual)

            if self.use_cache:
                self._cache[cache_key] = fitness
                self._cache_misses += 1

            return fitness
        except Exception as e:
            raise e

    def _get_cache_key(self, individual: Any) -> str:
        """キャッシュキーを生成"""
        if hasattr(individual, 'id') and individual.id:
            return str(individual.id)

        # IDがない場合は文字列表現を使用
        return str(individual)

    def get_statistics(self) -> Dict[str, Any]:
        """評価統計を取得"""
        total = self._cache_hits + self._cache_misses
        cache_hit_rate = self._cache_hits / total if total > 0 else 0.0

        return {
            "total_evaluations": self._total_evaluations,
            "successful_evaluations": self._successful_evaluations,
            "failed_evaluations": self._failed_evaluations,
            "timeout_evaluations": self._timeout_evaluations,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._cache),
        }

    def clear_cache(self):
        """キャッシュをクリア"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def reset_statistics(self):
        """統計情報をリセット"""
        self._total_evaluations = 0
        self._successful_evaluations = 0
        self._failed_evaluations = 0
        self._timeout_evaluations = 0
        self._cache_hits = 0
        self._cache_misses = 0
