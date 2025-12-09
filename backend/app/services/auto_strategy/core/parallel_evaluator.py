"""
並列評価モジュール

個体群の適応度評価を並列化して実行時間を短縮します。
ThreadPoolExecutorを使用し、I/Oバウンドなバックテスト処理を効率化。
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ParallelEvaluator:
    """
    並列評価クラス

    ThreadPoolExecutorを使用して個体群の適応度評価を並列化します。
    バックテストはI/Oバウンドな処理が多いため、ThreadPoolExecutorが適切です。
    """

    def __init__(
        self,
        evaluate_func: Callable[[Any], Tuple[float, ...]],
        max_workers: Optional[int] = None,
        timeout_per_individual: float = 300.0,
    ):
        """
        初期化

        Args:
            evaluate_func: 個体を評価する関数（toolbox.evaluate相当）
            max_workers: 最大ワーカー数（Noneの場合はCPUコア数の2倍）
            timeout_per_individual: 個体あたりのタイムアウト秒数
        """
        self.evaluate_func = evaluate_func
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) * 2)
        self.timeout_per_individual = timeout_per_individual
        self._executor: Optional[ThreadPoolExecutor] = None

        # 評価統計
        self._total_evaluations = 0
        self._successful_evaluations = 0
        self._failed_evaluations = 0
        self._timeout_evaluations = 0

    def evaluate_population(
        self,
        population: List[Any],
        default_fitness: Optional[Tuple[float, ...]] = None,
    ) -> List[Tuple[float, ...]]:
        """
        個体群を並列評価

        Args:
            population: 評価対象の個体リスト
            default_fitness: 評価失敗時のデフォルトフィットネス値

        Returns:
            各個体のフィットネス値のリスト（入力と同じ順序）
        """
        if not population:
            return []

        # デフォルトフィットネス値の設定
        if default_fitness is None:
            default_fitness = (0.0,)

        population_size = len(population)
        results: List[Optional[Tuple[float, ...]]] = [None] * population_size
        index_map = {id(ind): i for i, ind in enumerate(population)}

        logger.info(f"並列評価開始: {population_size}個体, {self.max_workers}ワーカー")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # すべての個体を並列に評価
            future_to_individual = {
                executor.submit(self._evaluate_single, ind): ind for ind in population
            }

            for future in as_completed(
                future_to_individual,
                timeout=self.timeout_per_individual * population_size,
            ):
                individual = future_to_individual[future]
                index = index_map[id(individual)]
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

        # Noneが残っている場合はデフォルト値で埋める
        final_results = [r if r is not None else default_fitness for r in results]

        logger.info(
            f"並列評価完了: 成功={self._successful_evaluations}, "
            f"失敗={self._failed_evaluations}, タイムアウト={self._timeout_evaluations}"
        )

        return final_results

    def _evaluate_single(self, individual: Any) -> Tuple[float, ...]:
        """
        単一個体の評価

        Args:
            individual: 評価対象の個体

        Returns:
            フィットネス値のタプル
        """
        return self.evaluate_func(individual)

    def evaluate_invalid_individuals(
        self,
        population: List[Any],
        default_fitness: Optional[Tuple[float, ...]] = None,
    ) -> List[Tuple[Tuple[float, ...], Any]]:
        """
        適応度が無効な個体のみを並列評価

        Args:
            population: 個体群
            default_fitness: 評価失敗時のデフォルトフィットネス値

        Returns:
            (フィットネス値, 個体) のタプルリスト
        """
        # 無効な個体を抽出
        invalid_individuals = [ind for ind in population if not ind.fitness.valid]

        if not invalid_individuals:
            return []

        logger.debug(f"無効な個体数: {len(invalid_individuals)}/{len(population)}")

        # 並列評価
        fitnesses = self.evaluate_population(invalid_individuals, default_fitness)

        return list(zip(fitnesses, invalid_individuals))

    def get_statistics(self) -> dict:
        """
        評価統計を取得

        Returns:
            統計情報の辞書
        """
        return {
            "total_evaluations": self._total_evaluations,
            "successful_evaluations": self._successful_evaluations,
            "failed_evaluations": self._failed_evaluations,
            "timeout_evaluations": self._timeout_evaluations,
            "success_rate": (
                self._successful_evaluations / self._total_evaluations
                if self._total_evaluations > 0
                else 0.0
            ),
        }

    def reset_statistics(self) -> None:
        """評価統計をリセット"""
        self._total_evaluations = 0
        self._successful_evaluations = 0
        self._failed_evaluations = 0
        self._timeout_evaluations = 0


def create_parallel_map(
    evaluate_func: Callable[[Any], Tuple[float, ...]],
    max_workers: Optional[int] = None,
) -> Callable[[Callable, List[Any]], List[Tuple[float, ...]]]:
    """
    DEAPツールボックス用の並列mapファンクションを作成

    Args:
        evaluate_func: 評価関数
        max_workers: 最大ワーカー数

    Returns:
        toolbox.mapに登録可能な関数
    """
    evaluator = ParallelEvaluator(
        evaluate_func=evaluate_func,
        max_workers=max_workers,
    )

    def parallel_map(func: Callable, individuals: List[Any]) -> List[Tuple[float, ...]]:
        """
        並列map関数

        Note: funcは無視され、初期化時のevaluate_funcが使用されます。
        これはDEAPのtoolbox.mapインターフェースに合わせるためです。
        """
        return evaluator.evaluate_population(individuals)

    return parallel_map
