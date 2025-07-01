"""
進化演算子

遺伝的アルゴリズムの進化演算（交叉、突然変異、選択、エリート保存）を担当するモジュール。
"""

import logging
import time
from typing import List, Any, Dict

logger = logging.getLogger(__name__)


class EvolutionOperators:
    """
    進化演算子

    GA の進化演算（交叉、突然変異、選択、エリート保存）を担当します。
    """

    def __init__(self):
        """初期化"""
        self.performance_stats = {
            "crossover": {"total_time": 0.0, "total_calls": 0, "total_operations": 0},
            "mutation": {"total_time": 0.0, "total_calls": 0, "total_operations": 0},
            "selection": {"total_time": 0.0, "total_calls": 0, "total_operations": 0},
            "elitism": {"total_time": 0.0, "total_calls": 0, "total_operations": 0},
            "evaluation": {"total_time": 0.0, "total_calls": 0, "total_operations": 0},
        }

    def apply_elitism(
        self,
        population: List[Any],
        offspring: List[Any],
        elite_size: int,
    ) -> List[Any]:
        """
        エリート保存戦略を適用します。
        遺伝的アルゴリズムにおいて、最も適応度の高い個体 (エリート) を
        次世代に直接引き継ぐことで、優れた遺伝子情報が失われるのを防ぎ、
        収束を早める効果があります。

        Args:
            population: 親世代の個体群
            offspring: 子世代の個体群
            elite_size: エリートとして次世代に残す個体の数

        Returns:
            エリート保存を適用した新しい個体群
        """
        try:
            # 親世代の個体を適応度が高い順にソートし、上位 elite_size 個体をエリートとして選出
            population.sort(key=lambda x: x.fitness.values[0], reverse=True)
            elite = population[:elite_size]  # 最も優れた個体群

            # 子世代から残りの個体を選択 (エリートの数だけ子世代から減らす)
            offspring.sort(key=lambda x: x.fitness.values[0], reverse=True)
            remaining_size = len(population) - elite_size
            selected_offspring = offspring[:remaining_size]

            # エリートと選択された子世代を結合して新しい個体群を形成
            new_population = elite + selected_offspring

            return new_population

        except Exception as e:
            logger.error(f"エリート保存エラー: {e}", exc_info=True)
            logger.warning("エリート保存に失敗しました。元の個体群を返します。")
            # エラー時は元の個体群を返す
            return population

    def perform_crossover(
        self, offspring: List[Any], crossover_rate: float, toolbox
    ) -> List[Any]:
        """
        交叉を実行します。
        個体群の中からランダムにペアを選択し、交叉率に基づいて遺伝子を交換します。
        交叉後、子個体の適応度を無効化し、再評価を促します。

        Args:
            offspring: 子世代個体群 (交叉の対象)
            crossover_rate: 交叉を行う確率 (0.0から1.0)
            toolbox: DEAPツールボックス (mate関数を含む)

        Returns:
            交叉後の個体群
        """
        start_time = time.time()
        try:
            import random

            crossover_count = 0
            total_pairs = len(offspring) // 2

            # 個体群を2つずつペアにしてイテレート
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                # 交叉率に基づいて交叉を実行
                if random.random() < crossover_rate:
                    toolbox.mate(child1, child2)  # type: ignore # mate関数で遺伝子を交換
                    del child1.fitness.values  # 適応度を無効化
                    del child2.fitness.values  # 適応度を無効化
                    crossover_count += 1

            # パフォーマンス統計を更新
            execution_time = time.time() - start_time
            self.performance_stats["crossover"]["total_time"] += execution_time
            self.performance_stats["crossover"]["total_calls"] += 1
            self.performance_stats["crossover"]["total_operations"] += crossover_count

            # 交叉統計をログ出力
            crossover_rate_actual = (
                crossover_count / total_pairs if total_pairs > 0 else 0
            )
            logger.info(
                f"交叉実行完了: {crossover_count}/{total_pairs}ペア "
                f"(実行率: {crossover_rate_actual:.1%}, 設定率: {crossover_rate:.1%}, "
                f"実行時間: {execution_time:.3f}秒)"
            )

            return offspring  # 正常終了時も個体群を返す

        except Exception as e:
            execution_time = time.time() - start_time
            self.performance_stats["crossover"]["total_time"] += execution_time
            self.performance_stats["crossover"]["total_calls"] += 1
            logger.error(f"交叉エラー: {e}", exc_info=True)
            return offspring  # エラー時も個体群を返す

    def perform_mutation(
        self, offspring: List[Any], mutation_rate: float, toolbox
    ) -> List[Any]:
        """
        突然変異を実行します。
        個体群の各個体に対し、突然変異率に基づいて遺伝子にランダムな変更を加えます。
        突然変異後、個体の適応度を無効化し、再評価を促します。

        Args:
            offspring: 子世代個体群 (突然変異の対象)
            mutation_rate: 突然変異を行う確率 (0.0から1.0)
            toolbox: DEAPツールボックス (mutate関数を含む)

        Returns:
            突然変異後の個体群
        """
        start_time = time.time()
        try:
            import random

            mutation_count = 0
            total_individuals = len(offspring)

            # 個体群の各個体に対してイテレート
            for mutant in offspring:
                # 突然変異率に基づいて突然変異を実行
                if random.random() < mutation_rate:
                    toolbox.mutate(mutant)  # type: ignore # mutate関数で遺伝子を変更
                    del mutant.fitness.values  # 適応度を無効化
                    mutation_count += 1

            # パフォーマンス統計を更新
            execution_time = time.time() - start_time
            self.performance_stats["mutation"]["total_time"] += execution_time
            self.performance_stats["mutation"]["total_calls"] += 1
            self.performance_stats["mutation"]["total_operations"] += mutation_count

            # 突然変異統計をログ出力
            mutation_rate_actual = (
                mutation_count / total_individuals if total_individuals > 0 else 0
            )
            logger.info(
                f"突然変異実行完了: {mutation_count}/{total_individuals}個体 "
                f"(実行率: {mutation_rate_actual:.1%}, 設定率: {mutation_rate:.1%}, "
                f"実行時間: {execution_time:.3f}秒)"
            )

            return offspring  # 正常終了時も個体群を返す

        except Exception as e:
            execution_time = time.time() - start_time
            self.performance_stats["mutation"]["total_time"] += execution_time
            self.performance_stats["mutation"]["total_calls"] += 1
            logger.error(f"突然変異エラー: {e}", exc_info=True)
            return offspring  # エラー時も個体群を返す

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

            return offspring

        except Exception as e:
            logger.error(f"親選択エラー: {e}", exc_info=True)
            logger.warning("親選択に失敗しました。元の個体群を返します。")
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

            return offspring

        except Exception as e:
            logger.error(f"無効個体評価エラー: {e}", exc_info=True)
            logger.warning(
                "無効個体の評価に失敗しました。評価されていない個体群を返します。"
            )
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

    def get_performance_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        進化演算子のパフォーマンス統計を取得

        Returns:
            各演算子の実行時間、呼び出し回数、操作回数の統計
        """
        stats = {}
        for operation, data in self.performance_stats.items():
            avg_time_per_call = (
                data["total_time"] / data["total_calls"]
                if data["total_calls"] > 0
                else 0
            )
            avg_operations_per_call = (
                data["total_operations"] / data["total_calls"]
                if data["total_calls"] > 0
                else 0
            )

            stats[operation] = {
                "total_time": data["total_time"],
                "total_calls": data["total_calls"],
                "total_operations": data["total_operations"],
                "avg_time_per_call": avg_time_per_call,
                "avg_operations_per_call": avg_operations_per_call,
            }

        return stats

    def reset_performance_statistics(self):
        """パフォーマンス統計をリセット"""
        for operation in self.performance_stats:
            self.performance_stats[operation] = {
                "total_time": 0.0,
                "total_calls": 0,
                "total_operations": 0,
            }
        logger.info("パフォーマンス統計をリセットしました")
