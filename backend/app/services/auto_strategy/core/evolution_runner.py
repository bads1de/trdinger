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

logger = logging.getLogger(__name__)


class EvolutionRunner:
    """
    進化計算の実行を担当するクラス

    単一目的と多目的最適化のロジックをカプセル化した独立クラス。
    GAエンジンから分離することで再利用性とテスト容易性を向上。
    """

    def __init__(
        self,
        toolbox,
        stats,
        fitness_sharing: Optional[FitnessSharing] = None,
        population: Optional[List[Any]] = None
    ):
        """
        初期化

        Args:
            toolbox: DEAPツールボックス
            stats: 統計情報収集オブジェクト
            fitness_sharing: 適応度共有オブジェクト（オプション）
            population: 個体集団（適応的突然変異用）
        """
        self.toolbox = toolbox
        self.stats = stats
        self.fitness_sharing = fitness_sharing
        self.population = population  # 適応的突然変異用

    def run_single_objective_evolution(
        self, population: List[Any], config: Any, halloffame: Optional[List[Any]] = None
    ) -> tuple[List[Any], Any]:
        """
        単一目的最適化アルゴリズムの実行

        Args:
            population: 初期個体群
            config: GA設定
            halloffame: 殿堂入り個体リスト

        Returns:
            (最終個体群, 進化ログ)
        """
        logger.info("単一目的最適化アルゴリズムを開始")

        # 初期適応度評価
        population = self._evaluate_population(population)
        self._update_dynamic_objective_scalars(population, config)

        logbook = tools.Logbook()

        # カスタム世代ループ（fitness_sharingを世代毎に適用）
        for gen in range(config.generations):
            logger.debug(f"世代 {gen + 1}/{config.generations} を開始")

            # 適応度共有の適用（有効な場合、世代毎）
            if getattr(config, 'enable_fitness_sharing', False) and self.fitness_sharing:
                population = self.fitness_sharing.apply_fitness_sharing(population)

            # 選択
            offspring = list(self.toolbox.map(self.toolbox.clone, population))
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

            # 評価
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 次世代の選択 (mu+lambda)
            population[:] = self.toolbox.select(offspring + population, len(population))

            self._update_dynamic_objective_scalars(population, config)

            # 統計の記録
            record = self.stats.compile(population) if self.stats else {}
            logbook.record(gen=gen, **record)

            # Hall of Fameの更新
            if halloffame is not None:
                halloffame.update(population)

        logger.info("単一目的最適化アルゴリズム完了")
        return population, logbook

    def run_multi_objective_evolution(
        self, population: List[Any], config: Any, halloffame: Optional[List[Any]] = None
    ) -> tuple[List[Any], Any]:
        """
        多目的最適化アルゴリズムの実行

        Args:
            population: 初期個体群
            config: GA設定
            halloffame: 殿堂入り個体リスト

        Returns:
            (最終個体群, 進化ログ)
        """
        logger.info("多目的最適化アルゴリズム（NSGA-II）を開始")

        # 初期適応度評価
        population = self._evaluate_population(population)
        self._update_dynamic_objective_scalars(population, config)

        # 多目的最適化用の選択関数に切り替え
        original_select = self.toolbox.select
        self.toolbox.select = tools.selNSGA2

        # パレートフロント更新
        pareto_front = tools.ParetoFront()
        population = self.toolbox.select(population, len(population))

        logbook = tools.Logbook()

        # カスタム世代ループ（fitness_sharingを世代毎に適用）
        for gen in range(config.generations):
            logger.debug(f"多目的世代 {gen + 1}/{config.generations} を開始")

            # 適応度共有の適用（有効な場合、世代毎）
            if getattr(config, 'enable_fitness_sharing', False) and self.fitness_sharing:
                population = self.fitness_sharing.apply_fitness_sharing(population)

            # 選択
            offspring = list(self.toolbox.map(self.toolbox.clone, population))
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

            # 評価
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 次世代の選択 (mu+lambda, NSGA-II)
            population[:] = self.toolbox.select(offspring + population, len(population))

            self._update_dynamic_objective_scalars(population, config)

            # 統計の記録
            record = self.stats.compile(population) if self.stats else {}
            logbook.record(gen=gen, **record)

            # Hall of Fameの更新
            if halloffame is not None:
                halloffame.update(population)

        # パレートフロントを更新
        for ind in population:
            pareto_front.update(population)

        # 選択関数を元に戻す
        self.toolbox.select = original_select

        logger.info("多目的最適化アルゴリズム（NSGA-II）完了")
        return population, logbook

    def _evaluate_population(self, population: List[Any]) -> List[Any]:
        """
        個体群の適応度評価

        Args:
            population: 評価対象の個体群

        Returns:
            評価された個体群
        """
        # 初期個体群の評価
        fitnesses = list(self.toolbox.map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        return population

    def _update_dynamic_objective_scalars(self, population: List[Any], config: Any) -> None:
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
