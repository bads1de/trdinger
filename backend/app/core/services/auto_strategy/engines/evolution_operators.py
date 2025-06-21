"""
進化演算子

遺伝的アルゴリズムの進化演算（交叉、突然変異、選択、エリート保存）を担当するモジュール。
"""

import logging
from typing import List, Any

logger = logging.getLogger(__name__)


class EvolutionOperators:
    """
    進化演算子

    GA の進化演算（交叉、突然変異、選択、エリート保存）を担当します。
    """

    def __init__(self):
        """初期化"""
        pass

    def apply_elitism(
        self,
        population: List[Any],
        offspring: List[Any],
        elite_size: int,
    ) -> List[Any]:
        """
        エリート保存戦略を適用

        遺伝的アルゴリズムにおいて、最も適応度の高い個体 (エリート) を
        次世代に直接引き継ぐことで、優れた遺伝子情報が失われるのを防ぎ、
        収束を早める効果があります。

        Args:
            population: 親世代の個体群
            offspring: 子世代の個体群
            elite_size: エリートサイズ

        Returns:
            エリート保存を適用した新しい個体群
        """
        try:
            # 親世代の個体を適応度が高い順にソートし、上位 elite_size 個体をエリートとして選出
            population.sort(key=lambda x: x.fitness.values[0], reverse=True)
            elite = population[:elite_size]  # 最も優れた個体群

            # 子世代から残りの個体を選択
            offspring.sort(key=lambda x: x.fitness.values[0], reverse=True)
            remaining_size = len(population) - elite_size
            selected_offspring = offspring[:remaining_size]

            # エリートと選択された子世代を結合
            new_population = elite + selected_offspring

            logger.debug(
                f"エリート保存適用: エリート{elite_size}個体 + 子世代{remaining_size}個体"
            )
            return new_population

        except Exception as e:
            logger.error(f"エリート保存エラー: {e}")
            # エラー時は元の個体群を返す
            return population

    def perform_crossover(
        self, offspring: List[Any], crossover_rate: float, toolbox
    ) -> List[Any]:
        """
        交叉を実行

        Args:
            offspring: 子世代個体群
            crossover_rate: 交叉率
            toolbox: DEAPツールボックス

        Returns:
            交叉後の個体群
        """
        try:
            import random

            crossover_count = 0
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < crossover_rate:
                    toolbox.mate(child1, child2)  # type: ignore
                    del child1.fitness.values
                    del child2.fitness.values
                    crossover_count += 1

            logger.debug(f"交叉実行: {crossover_count}組の個体で交叉")
            return offspring

        except Exception as e:
            logger.error(f"交叉エラー: {e}")
            return offspring

    def perform_mutation(
        self, offspring: List[Any], mutation_rate: float, toolbox
    ) -> List[Any]:
        """
        突然変異を実行

        Args:
            offspring: 子世代個体群
            mutation_rate: 突然変異率
            toolbox: DEAPツールボックス

        Returns:
            突然変異後の個体群
        """
        try:
            import random

            mutation_count = 0
            for mutant in offspring:
                if random.random() < mutation_rate:
                    toolbox.mutate(mutant)  # type: ignore
                    del mutant.fitness.values
                    mutation_count += 1

            logger.debug(f"突然変異実行: {mutation_count}個体で突然変異")
            return offspring

        except Exception as e:
            logger.error(f"突然変異エラー: {e}")
            return offspring

    def select_parents(self, population: List[Any], toolbox) -> List[Any]:
        """
        親個体を選択

        Args:
            population: 現在の個体群
            toolbox: DEAPツールボックス

        Returns:
            選択された親個体群
        """
        try:
            # 選択
            offspring = toolbox.select(population, len(population))  # type: ignore
            offspring = list(map(toolbox.clone, offspring))  # type: ignore

            logger.debug(f"親選択完了: {len(offspring)}個体を選択")
            return offspring

        except Exception as e:
            logger.error(f"親選択エラー: {e}")
            return population

    def evaluate_invalid_individuals(self, offspring: List[Any], toolbox) -> List[Any]:
        """
        無効な個体を評価

        Args:
            offspring: 子世代個体群
            toolbox: DEAPツールボックス

        Returns:
            評価済みの個体群
        """
        try:
            # 無効な個体の評価
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            if invalid_ind:
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)  # type: ignore
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                logger.debug(f"無効個体評価完了: {len(invalid_ind)}個体を評価")

            return offspring

        except Exception as e:
            logger.error(f"無効個体評価エラー: {e}")
            return offspring

    def get_fitness_statistics(self, population: List[Any]) -> dict:
        """
        個体群の適応度統計を取得

        Args:
            population: 個体群

        Returns:
            適応度統計情報
        """
        try:
            fitnesses = [
                ind.fitness.values[0] for ind in population if ind.fitness.valid
            ]

            if not fitnesses:
                return {"max": 0.0, "min": 0.0, "avg": 0.0, "count": 0}

            return {
                "max": max(fitnesses),
                "min": min(fitnesses),
                "avg": sum(fitnesses) / len(fitnesses),
                "count": len(fitnesses),
            }

        except Exception as e:
            logger.error(f"適応度統計計算エラー: {e}")
            return {"max": 0.0, "min": 0.0, "avg": 0.0, "count": 0}
