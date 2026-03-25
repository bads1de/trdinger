"""
進化計算実行モジュール

進化計算の実行を担当するEvolutionRunnerクラスを提供します。
単一目的と多目的最適化のロジックをカプセル化し、並列評価をサポートします。
"""

import gc
import logging
import random
from typing import Any, List, Optional

import numpy as np
from deap import tools

from ..fitness.fitness_sharing import FitnessSharing
from .ga_utils import _invalidate_individual_cache, _set_fitness_values
from ..evaluation.parallel_evaluator import ParallelEvaluator

logger = logging.getLogger(__name__)


class EvolutionRunner:
    """
    進化計算の実行を担当するクラス

    単一目的と多目的最適化のロジックをカプセル化した独立クラス。
    GAエンジンから分割されたが、現在は同一モジュール内で定義。
    並列評価をサポート。
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

    def run_evolution(
        self, population: List[Any], config: Any, halloffame: Optional[Any] = None
    ) -> tuple[List[Any], Any]:
        """
        進化アルゴリズムの実行（単一・多目的 統一版）

        単一目的・多目的を問わず、toolboxに登録された演算子と
        渡されたhalloffameオブジェクト（HallOfFame または ParetoFront）を使用して
        進化計算を実行します。

        Args:
            population: 初期個体群
            config: GA設定
            halloffame: 殿堂入り個体リスト（HallOfFame または ParetoFront）

        Returns:
            (最終個体群, 進化ログ)
        """
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
                # cloneを使用することで、交叉・変異が元の個体に影響しないようにする
                offspring = list(self.toolbox.map(self.toolbox.clone, population))

                # 交叉
                # インデックスを使ってリストを直接更新する（mateが新しいオブジェクトを返すため）
                for i in range(0, len(offspring) - 1, 2):
                    child1, child2 = offspring[i], offspring[i + 1]
                    if random.random() < config.crossover_rate:
                        new_child1, new_child2 = self.toolbox.mate(child1, child2)
                        offspring[i] = new_child1
                        offspring[i + 1] = new_child2
                        _invalidate_individual_cache(offspring[i])
                        _invalidate_individual_cache(offspring[i + 1])

                # 突然変異
                # インデックスを使ってリストを直接更新する（mutateが新しいオブジェクトを返すため）
                for i in range(len(offspring)):
                    mutant = offspring[i]
                    if random.random() < config.mutation_rate:
                        # mutateはタプル(ind,)を返す
                        result = self.toolbox.mutate(mutant)
                        new_mutant = result[0]
                        offspring[i] = new_mutant
                        _invalidate_individual_cache(offspring[i])

                # 未評価個体の評価（並列評価対応）
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                self._evaluate_invalid_individuals(invalid_ind)

                # 次世代の選択 (mu+lambda)
                # toolbox.select は DEAPSetup で NSGA-II などが登録されている
                population[:] = self.toolbox.select(
                    offspring + population, len(population)
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

    def _evaluate_population(self, population: List[Any]) -> List[Any]:
        """
        個体群の適応度評価（並列評価対応）

        ツールボックスに登録された評価関数を使用して、
        集団内の全個体の適応度を計算します。並列評価器が利用可能な場合は、
        複数のプロセスで並列に評価を実行し、計算時間を短縮します。

        Args:
            population: 評価対象の個体群

        Returns:
            評価値（fitness.values）が設定された個体群
        """
        if self.parallel_evaluator:
            # 並列評価
            fitnesses = self.parallel_evaluator.evaluate_population(population)
            _set_fitness_values(population, fitnesses)
        else:
            # シーケンシャル評価（フォールバック）
            fitnesses = list(self.toolbox.map(self.toolbox.evaluate, population))
            _set_fitness_values(population, fitnesses)

        return population

    def _evaluate_invalid_individuals(self, invalid_ind: List[Any]) -> None:
        """
        適応度が無効な個体のみを評価（並列評価対応）

        Args:
            invalid_ind: 評価対象の無効な個体リスト
        """
        if not invalid_ind:
            return

        if self.parallel_evaluator:
            # 並列評価
            fitnesses = self.parallel_evaluator.evaluate_population(invalid_ind)
            _set_fitness_values(invalid_ind, fitnesses)
        else:
            # シーケンシャル評価（フォールバック）
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            _set_fitness_values(invalid_ind, fitnesses)

    def _update_dynamic_objective_scalars(
        self, population: List[Any], config: Any
    ) -> None:
        """
        リスク回避型の重み付けのために、動的な目的正規化係数を更新します。

        集団全体の平均的なパフォーマンスに基づいて、特定の指標（ドローダウンなど）の
        ペナルティやウェイトを調整します。

        Args:
            population: 現在の集団
            config: GA設定
        """

        if not getattr(config, "dynamic_objective_reweighting", False):
            config.objective_dynamic_scalars = {}
            return

        if not population:
            config.objective_dynamic_scalars = {}
            return

        scalars: dict[str, float] = {}
        for index, objective in enumerate(getattr(config, "objectives", [])):
            values: List[float] = []
            for individual in population:
                fitness = getattr(individual, "fitness", None)
                if not fitness or not getattr(fitness, "valid", False):
                    continue
                fitness_values = getattr(fitness, "values", ())
                if len(fitness_values) <= index:
                    continue
                try:
                    values.append(float(fitness_values[index]))
                except (TypeError, ValueError):
                    continue

            if not values:
                continue

            average_value = float(np.mean(values))
            if objective in {"max_drawdown", "ulcer_index", "trade_frequency_penalty"}:
                scalars[objective] = min(2.0, 1.0 + max(average_value, 0.0))
            else:
                scalars[objective] = 1.0

        config.objective_dynamic_scalars = scalars
