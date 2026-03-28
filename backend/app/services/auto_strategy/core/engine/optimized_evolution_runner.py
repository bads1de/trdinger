"""
最適化された進化計算実行モジュール

進化計算の実行を担当する最適化されたEvolutionRunnerクラスを提供します。
主な最適化ポイント:
1. 交叉・突然変異のバッチ処理
2. 未評価個体のフィルタリング最適化
3. メモリ効率的な個体管理
"""

import gc
import logging
import random
from typing import Any, Callable, Dict, List, Optional


from deap import tools

from app.services.auto_strategy.core.fitness.fitness_sharing import FitnessSharing
from app.services.auto_strategy.core.engine.ga_utils import (
    _invalidate_individual_cache,
    _set_fitness_values,
)
from app.services.auto_strategy.core.evaluation.parallel_evaluator import (
    ParallelEvaluator,
)

logger = logging.getLogger(__name__)


class EvolutionStoppedError(RuntimeError):
    """進化処理が停止要求によって中断されたことを示す例外"""


class OptimizedEvolutionRunner:
    """
    最適化された進化計算の実行を担当するクラス

    主な最適化ポイント:
    1. 交叉・突然変異のバッチ処理
    2. 未評価個体のフィルタリング最適化
    3. メモリ効率的な個体管理
    4. 統計情報の効率的な収集
    """

    def __init__(
        self,
        toolbox,
        stats,
        fitness_sharing: Optional[FitnessSharing] = None,
        population: Optional[List[Any]] = None,
        parallel_evaluator: Optional[ParallelEvaluator] = None,
    ):
        """
        初期化

        Args:
            toolbox: DEAPツールボックス
            stats: 統計情報収集オブジェクト
            fitness_sharing: 適応度共有オブジェクト（オプション）
            population: 個体集団（適応的突然変異用）
            parallel_evaluator: 並列評価器（オプション）
        """
        self.toolbox = toolbox
        self.stats = stats
        self.fitness_sharing = fitness_sharing
        self.population = population  # 適応的突然変異用
        self.parallel_evaluator = parallel_evaluator

        # 最適化: 交叉・突然変異のキャッシュ
        self._crossover_cache: Dict[str, Any] = {}
        self._mutation_cache: Dict[str, Any] = {}

    def run_evolution(
        self,
        population: List[Any],
        config: Any,
        halloffame: Optional[Any] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> tuple[List[Any], Any]:
        """
        進化アルゴリズムの実行（単一・多目的 統一版）

        Args:
            population: 初期個体群
            config: GA設定
            halloffame: 殿堂入り個体リスト（HallOfFame または ParetoFront）

        Returns:
            (最終個体群, 進化ログ)
        """
        if should_stop and should_stop():
            raise EvolutionStoppedError("進化処理は開始前に停止されました")

        logger.info(
            f"進化アルゴリズムを開始（世代数: {config.generations}, 目的数: {len(config.objectives)}）"
        )

        # 初期適応度評価
        population = self._evaluate_population(population)
        self._update_dynamic_objective_scalars(population, config)

        # Hall of Fame / Pareto Front 初回更新
        if halloffame is not None:
            halloffame.update(population)

        logbook = tools.Logbook()

        # 世代ループ
        for gen in range(config.generations):
            if should_stop and should_stop():
                raise EvolutionStoppedError(
                    f"進化処理は世代 {gen + 1} の開始前に停止されました"
                )

            # GC制御: 世代の変わり目でまとめて回収し、計算中の停止を防ぐ
            gc.collect()
            gc.disable()

            try:
                logger.debug("世代 %s/%s を開始", gen + 1, config.generations)

                # 適応度共有の適用（有効な場合、世代毎）
                if (
                    getattr(config, "enable_fitness_sharing", False)
                    and self.fitness_sharing
                ):
                    population = self.fitness_sharing.apply_fitness_sharing(population)

                # 選択（親個体の選択）
                offspring = list(self.toolbox.map(self.toolbox.clone, population))

                # 最適化: 交叉のバッチ処理
                offspring = self._apply_crossover_batch(offspring, config)

                # 最適化: 突然変異のバッチ処理
                offspring = self._apply_mutation_batch(offspring, config)

                # 未評価個体の評価（並列評価対応）
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                self._evaluate_invalid_individuals(invalid_ind)

                if should_stop and should_stop():
                    raise EvolutionStoppedError(
                        f"進化処理は世代 {gen + 1} の評価後に停止されました"
                    )

                # 次世代の選択 (mu+lambda)
                population[:] = self.toolbox.select(
                    offspring + population, len(population)
                )

                if should_stop and should_stop():
                    raise EvolutionStoppedError(
                        f"進化処理は世代 {gen + 1} の選択後に停止されました"
                    )

                self._update_dynamic_objective_scalars(population, config)

                # 統計の記録
                record = self.stats.compile(population) if self.stats else {}
                logbook.record(gen=gen, **record)

                # Hall of Fame / Pareto Front の更新
                if halloffame is not None:
                    halloffame.update(population)
            finally:
                # ループ終了時にGCを有効化（エラー時も含む）
                gc.enable()

        logger.info("進化アルゴリズム完了")
        return population, logbook

    def _apply_crossover_batch(self, offspring: List[Any], config: Any) -> List[Any]:
        """
        交叉のバッチ処理（最適化版）

        最適化:
        - インデックスを使ってリストを直接更新
        - 交叉キャッシュの活用
        """
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < config.crossover_rate:
                child1, child2 = offspring[i], offspring[i + 1]

                # キャッシュチェック
                cache_key = self._get_crossover_cache_key(child1, child2)
                if cache_key in self._crossover_cache:
                    new_child1, new_child2 = self._crossover_cache[cache_key]
                else:
                    new_child1, new_child2 = self.toolbox.mate(child1, child2)
                    self._crossover_cache[cache_key] = (new_child1, new_child2)

                offspring[i] = new_child1
                offspring[i + 1] = new_child2
                _invalidate_individual_cache(offspring[i])
                _invalidate_individual_cache(offspring[i + 1])

        return offspring

    def _apply_mutation_batch(self, offspring: List[Any], config: Any) -> List[Any]:
        """
        突然変異のバッチ処理（最適化版）

        最適化:
        - インデックスを使ってリストを直接更新
        - 突然変異キャッシュの活用
        """
        for i in range(len(offspring)):
            if random.random() < config.mutation_rate:
                mutant = offspring[i]

                # キャッシュチェック
                cache_key = self._get_mutation_cache_key(mutant)
                if cache_key in self._mutation_cache:
                    new_mutant = self._mutation_cache[cache_key]
                else:
                    result = self.toolbox.mutate(mutant)
                    new_mutant = result[0]
                    self._mutation_cache[cache_key] = new_mutant

                offspring[i] = new_mutant
                _invalidate_individual_cache(offspring[i])

        return offspring

    def _get_crossover_cache_key(self, parent1: Any, parent2: Any) -> str:
        """交叉キャッシュキーを生成"""
        try:
            p1_id = getattr(parent1, "id", "") or str(id(parent1))
            p2_id = getattr(parent2, "id", "") or str(id(parent2))
            return f"{p1_id}:{p2_id}"
        except Exception:
            return str(id(parent1)) + ":" + str(id(parent2))

    def _get_mutation_cache_key(self, individual: Any) -> str:
        """突然変異キャッシュキーを生成"""
        try:
            ind_id = getattr(individual, "id", "") or str(id(individual))
            return ind_id
        except Exception:
            return str(id(individual))

    def _evaluate_population(self, population: List[Any]) -> List[Any]:
        """
        個体群の適応度評価（並列評価対応）

        Args:
            population: 評価対象の個体群

        Returns:
            評価値（fitness.values）が設定された個体群
        """
        if self.parallel_evaluator:
            fitnesses = self.parallel_evaluator.evaluate_population(population)
            _set_fitness_values(population, fitnesses)
        else:
            fitnesses = self.toolbox.map(self.toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

        return population

    def _evaluate_invalid_individuals(self, invalid_individuals: List[Any]) -> None:
        """
        未評価個体のみを評価（最適化版）

        Args:
            invalid_individuals: 未評価個体のリスト
        """
        if not invalid_individuals:
            return

        if self.parallel_evaluator:
            fitnesses = self.parallel_evaluator.evaluate_population(invalid_individuals)
            _set_fitness_values(invalid_individuals, fitnesses)
        else:
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_individuals)
            for ind, fit in zip(invalid_individuals, fitnesses):
                ind.fitness.values = fit

    def _update_dynamic_objective_scalars(
        self, population: List[Any], config: Any
    ) -> None:
        """
        動的目的スカラーの更新

        Args:
            population: 個体群
            config: GA設定
        """
        if not getattr(config, "dynamic_objective_reweighting", False):
            return

        try:
            from app.services.auto_strategy.core.fitness.objective_weights import (
                ObjectiveWeights,
            )

            ObjectiveWeights.update_scalars(population, config)
        except Exception as e:
            logger.debug(f"動的目的スカラー更新エラー: {e}")

    def clear_caches(self):
        """キャッシュをクリア"""
        self._crossover_cache.clear()
        self._mutation_cache.clear()
