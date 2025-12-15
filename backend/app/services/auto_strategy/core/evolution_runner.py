"""
進化計算実行モジュール

単一目的と多目的進化アルゴリズムの実行を担当する独立モジュール。
GAエンジンからの分割により、責務の明確化とテスト容易性を向上。
"""

import logging
import random
from typing import Any, Dict, List, Optional

import numpy as np
from deap import tools

from .fitness_sharing import FitnessSharing
from .parallel_evaluator import ParallelEvaluator

logger = logging.getLogger(__name__)


class EvolutionRunner:
    """
    進化計算の実行を担当するクラス

    単一目的と多目的最適化のロジックをカプセル化した独立クラス。
    GAエンジンから分離することで再利用性とテスト容易性を向上。
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
            logger.debug(f"世代 {gen + 1}/{config.generations} を開始")

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
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < config.crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # 突然変異
            for mutant in offspring:
                if random.random() < config.mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # 未評価個体の評価（並列評価対応）
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            self._evaluate_invalid_individuals(invalid_ind)

            # 次世代の選択 (mu+lambda)
            # toolbox.select は DEAPSetup で NSGA-II などが登録されている
            population[:] = self.toolbox.select(offspring + population, len(population))

            self._update_dynamic_objective_scalars(population, config)

            # 統計の記録
            record = self.stats.compile(population) if self.stats else {}
            logbook.record(gen=gen, **record)

            # Hall of Fame / Pareto Front の更新
            if halloffame is not None:
                halloffame.update(population)

        logger.info("進化アルゴリズム完了")
        return population, logbook

    def _evaluate_population(self, population: List[Any]) -> List[Any]:
        """
        個体群の適応度評価（並列評価対応）

        Args:
            population: 評価対象の個体群

        Returns:
            評価された個体群
        """
        if self.parallel_evaluator:
            # 並列評価
            fitnesses = self.parallel_evaluator.evaluate_population(population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
        else:
            # シーケンシャル評価（フォールバック）
            fitnesses = list(self.toolbox.map(self.toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

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
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
        else:
            # シーケンシャル評価（フォールバック）
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

    def _update_dynamic_objective_scalars(
        self, population: List[Any], config: Any
    ) -> None:
        """Update dynamic objective scaling factors for risk-aware weighting."""

        if not getattr(config, "dynamic_objective_reweighting", False):
            config.objective_dynamic_scalars = {}
            return

        if not population:
            config.objective_dynamic_scalars = {}
            return

        scalars: Dict[str, float] = {}
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





