"""
並列評価モジュール

個体群の適応度評価を並列化して実行時間を短縮します。
ProcessPoolExecutor を使用してタイムアウト時のゾンビプロセス問題を回避し、
CPU バウンドな計算を効率化します。
"""

import logging
import os
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    as_completed,
)
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# プロセスプール用のトップレベル評価関数（pickle可能にするため）
_global_evaluate_func: Optional[Callable] = None


def _init_worker(evaluate_func: Callable) -> None:
    """ワーカープロセス初期化関数"""
    global _global_evaluate_func
    _global_evaluate_func = evaluate_func


def _evaluate_in_worker(individual: Any) -> Tuple[float, ...]:
    """ワーカープロセス内で評価を実行"""
    global _global_evaluate_func
    if _global_evaluate_func is None:
        raise RuntimeError("評価関数が初期化されていません")
    return _global_evaluate_func(individual)


class ParallelEvaluator:
    """
    並列評価クラス

    ProcessPoolExecutor または ThreadPoolExecutor を使用して
    個体群の適応度評価を並列化します。

    ProcessPoolExecutor を使用することで、タイムアウト時に
    ワーカープロセスを強制終了でき、ゾンビタスク問題を回避できます。
    """

    def __init__(
        self,
        evaluate_func: Callable[[Any], Tuple[float, ...]],
        max_workers: Optional[int] = None,
        timeout_per_individual: float = 300.0,
        use_process_pool: bool = False,
        worker_initializer: Optional[Callable] = None,
        worker_initargs: Tuple = (),
    ):
        """
        初期化

        Args:
            evaluate_func: 個体を評価する関数（toolbox.evaluate相当）
            max_workers: 最大ワーカー数（Noneの場合はCPUコア数）
            timeout_per_individual: 個体あたりのタイムアウト秒数
            use_process_pool: ProcessPoolExecutorを使用するか
            worker_initializer: ワーカープロセス初期化関数（ProcessPool用）
            worker_initargs: 初期化関数の引数
        """
        self.evaluate_func = evaluate_func
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) * 2)
        self.timeout_per_individual = timeout_per_individual
        self.use_process_pool = use_process_pool
        self.worker_initializer = worker_initializer
        self.worker_initargs = worker_initargs

        # 評価統計
        self._total_evaluations = 0
        self._successful_evaluations = 0
        self._failed_evaluations = 0
        self._timeout_evaluations = 0

        # エラー種別の詳細統計
        self._error_categories: dict = {
            "timeout": 0,
            "memory": 0,
            "data_error": 0,
            "logic_error": 0,
            "other": 0,
        }

        # 最近のエラー履歴（デバッグ用、最大20件）
        self._recent_errors: list = []
        self._max_error_history = 20

        # 世代ごとの統計リセットフラグ
        self._auto_reset_per_generation = True

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

        executor_type = "Process" if self.use_process_pool else "Thread"
        logger.info(
            f"並列評価開始: {population_size}個体, {self.max_workers}ワーカー ({executor_type}Pool)"
        )

        # 世代開始時に統計をリセット
        if self._auto_reset_per_generation:
            self._reset_generation_stats()

        # Executor選択
        if self.use_process_pool:
            results = self._evaluate_with_process_pool(
                population, results, index_map, default_fitness
            )
        else:
            results = self._evaluate_with_thread_pool(
                population, results, index_map, default_fitness
            )

        # Noneが残っている場合はデフォルト値で埋める
        final_results = [r if r is not None else default_fitness for r in results]

        logger.info(
            f"並列評価完了: 成功={self._successful_evaluations}, "
            f"失敗={self._failed_evaluations}, タイムアウト={self._timeout_evaluations}"
        )

        return final_results

    def _evaluate_with_thread_pool(
        self,
        population: List[Any],
        results: List[Optional[Tuple[float, ...]]],
        index_map: dict,
        default_fitness: Tuple[float, ...],
    ) -> List[Optional[Tuple[float, ...]]]:
        """ThreadPoolExecutorを使用した評価"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_individual = {
                executor.submit(self._evaluate_single, ind): ind for ind in population
            }

            # 全体のタイムアウトを設定
            total_timeout = self.timeout_per_individual * len(population)

            try:
                for future in as_completed(future_to_individual, timeout=total_timeout):
                    individual = future_to_individual[future]
                    index = index_map[id(individual)]
                    self._total_evaluations += 1

                    try:
                        fitness = future.result(timeout=self.timeout_per_individual)
                        results[index] = fitness
                        self._successful_evaluations += 1
                    except FuturesTimeoutError:
                        logger.warning(f"個体評価タイムアウト: index={index}")
                        results[index] = default_fitness
                        self._timeout_evaluations += 1
                        self._error_categories["timeout"] += 1
                    except Exception as e:
                        category = self._categorize_error(e, index)
                        logger.warning(
                            f"個体評価エラー: index={index}, "
                            f"category={category}, error={e}"
                        )
                        results[index] = default_fitness
                        self._failed_evaluations += 1
            except FuturesTimeoutError:
                logger.warning("全体タイムアウト: 一部の個体が評価されませんでした")
                # 未完了のfutureをキャンセル
                for future in future_to_individual:
                    if not future.done():
                        future.cancel()
                        self._timeout_evaluations += 1
                        self._error_categories["timeout"] += 1

        return results

    def _evaluate_with_process_pool(
        self,
        population: List[Any],
        results: List[Optional[Tuple[float, ...]]],
        index_map: dict,
        default_fitness: Tuple[float, ...],
    ) -> List[Optional[Tuple[float, ...]]]:
        """ProcessPoolExecutorを使用した評価（ゾンビ対策済み）"""
        # ProcessPoolExecutorは子プロセスを強制終了できるため、
        # タイムアウト時にゾンビプロセスが発生しない

        # 内部初期化関数とユーザー指定初期化関数を組み合わせる
        def _combined_initializer(init_func, init_args, eval_func):
            # まず評価関数をセット
            _init_worker(eval_func)
            # 次にユーザー指定の初期化を実行
            if init_func:
                init_func(*init_args)

        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=_combined_initializer,
            initargs=(
                self.worker_initializer,
                self.worker_initargs,
                self.evaluate_func,
            ),
        ) as executor:
            future_to_index = {}

            for i, ind in enumerate(population):
                # 個体データをシリアライズして渡す
                future = executor.submit(self.evaluate_func, ind)
                future_to_index[future] = i

            total_timeout = self.timeout_per_individual * len(population)

            try:
                for future in as_completed(future_to_index, timeout=total_timeout):
                    index = future_to_index[future]
                    self._total_evaluations += 1

                    try:
                        fitness = future.result(timeout=self.timeout_per_individual)
                        results[index] = fitness
                        self._successful_evaluations += 1
                    except FuturesTimeoutError:
                        logger.warning(f"個体評価タイムアウト: index={index}")
                        results[index] = default_fitness
                        self._timeout_evaluations += 1
                        self._error_categories["timeout"] += 1
                    except Exception as e:
                        category = self._categorize_error(e, index)
                        logger.warning(
                            f"個体評価エラー: index={index}, "
                            f"category={category}, error={e}"
                        )
                        results[index] = default_fitness
                        self._failed_evaluations += 1
            except FuturesTimeoutError:
                logger.warning("全体タイムアウト: プロセスプールをシャットダウン")
                # ProcessPoolExecutorはコンテキスト終了時に子プロセスを終了
                timeout_count = sum(1 for r in results if r is None)
                self._timeout_evaluations += timeout_count
                self._error_categories["timeout"] += timeout_count

        return results

    def _evaluate_single(self, individual: Any) -> Tuple[float, ...]:
        """
        単一個体の評価

        Args:
            individual: 評価対象の個体

        Returns:
            フィットネス値のタプル
        """
        return self.evaluate_func(individual)

    def _categorize_error(self, error: Exception, index: int) -> str:
        """
        エラーを種別に分類

        Args:
            error: 発生した例外
            index: 個体のインデックス

        Returns:
            エラーカテゴリ名
        """
        error_type = type(error).__name__
        error_message = str(error).lower()

        # エラー種別の判定
        if isinstance(error, MemoryError) or "memory" in error_message:
            category = "memory"
        elif isinstance(error, (KeyError, IndexError, ValueError)):
            # データアクセスや変換に関連するエラー
            category = "data_error"
        elif isinstance(error, (TypeError, AttributeError)):
            # ロジックエラー（型不一致、属性不在など）
            category = "logic_error"
        elif isinstance(error, FuturesTimeoutError):
            category = "timeout"
        else:
            category = "other"

        # 統計更新
        self._error_categories[category] += 1

        # エラー履歴に追加
        error_info = {
            "index": index,
            "category": category,
            "error_type": error_type,
            "message": str(error)[:200],  # メッセージは200文字に制限
        }

        if len(self._recent_errors) >= self._max_error_history:
            self._recent_errors.pop(0)
        self._recent_errors.append(error_info)

        return category

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
            # エラー種別の詳細
            "error_categories": self._error_categories.copy(),
            # 最近のエラー履歴
            "recent_errors": self._recent_errors.copy(),
        }

    def reset_statistics(self) -> None:
        """評価統計をリセット"""
        self._total_evaluations = 0
        self._successful_evaluations = 0
        self._failed_evaluations = 0
        self._timeout_evaluations = 0
        # エラー種別もリセット
        for key in self._error_categories:
            self._error_categories[key] = 0
        self._recent_errors.clear()

    def _reset_generation_stats(self) -> None:
        """世代ごとの統計をリセット（成功/失敗/タイムアウトのみ）"""
        self._successful_evaluations = 0
        self._failed_evaluations = 0
        self._timeout_evaluations = 0


def create_parallel_map(
    evaluate_func: Callable[[Any], Tuple[float, ...]],
    max_workers: Optional[int] = None,
    use_process_pool: bool = False,
) -> Callable[[Callable, List[Any]], List[Tuple[float, ...]]]:
    """
    DEAPツールボックス用の並列mapファンクションを作成

    Args:
        evaluate_func: 評価関数
        max_workers: 最大ワーカー数
        use_process_pool: ProcessPoolExecutorを使用するか

    Returns:
        toolbox.mapに登録可能な関数
    """
    evaluator = ParallelEvaluator(
        evaluate_func=evaluate_func,
        max_workers=max_workers,
        use_process_pool=use_process_pool,
    )

    def parallel_map(func: Callable, individuals: List[Any]) -> List[Tuple[float, ...]]:
        """
        並列map関数

        Note: funcは無視され、初期化時のevaluate_funcが使用されます。
        これはDEAPのtoolbox.mapインターフェースに合わせるためです。
        """
        return evaluator.evaluate_population(individuals)

    return parallel_map
