"""
GA結果処理モジュール

進化計算の結果処理、最良個体の抽出、集団のランク付けなどの責務を担当します。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from deap import tools

from app.services.auto_strategy.genes import StrategyGene

from ..evaluation.evaluation_fidelity import is_multi_fidelity_enabled
from .report_selection import (
    extract_primary_fitness,
    get_two_stage_best_individual,
    get_two_stage_rank,
)

logger = logging.getLogger(__name__)


class ResultProcessor:
    """
    GA結果処理クラス

    進化計算の結果から最良個体を抽出し、集団をランク付けして
    永続化用のデータを準備します。
    """

    def extract_best_individuals(
        self,
        population: List[Any],
        config: Any,
        halloffame: Optional[Any] = None,
    ) -> Tuple[Any, Optional[StrategyGene], Optional[List[Dict[str, Any]]]]:
        """
        最終集団または殿堂入りオブジェクトから最良の個体群を抽出

        多目的最適化の場合はパレートフロントから、単一目的の場合は
        単純な最高スコア個体を選択し、バックテストでそのまま利用可能な
        形式に変換して返します。

        Args:
            population: 最終世代の全個体リスト
            config: GA 設定
            halloffame: 保存されている優良個体のリスト（またはパレートフロント）

        Returns:
            (最良個体, 最良遺伝子, 最良戦略リスト) のタプル
        """
        best_strategies = None
        best_individual = None
        best_gene = None

        if config.enable_multi_objective:
            # 多目的最適化の場合、パレート最適解を取得
            if halloffame is None or not isinstance(halloffame, tools.ParetoFront):
                pareto_front = tools.ParetoFront()
                pareto_front.update(population)
                best_individuals = list(pareto_front)
            else:
                best_individuals = list(halloffame)

            # 空の場合のガード
            if not best_individuals:
                best_individuals = [tools.selBest(population, 1)[0]]

            best_individual = best_individuals[0]

            best_strategies = []
            for ind in best_individuals[:10]:  # 上位10個のパレート最適解
                if isinstance(ind, StrategyGene):
                    gene = ind
                else:
                    logger.error(f"個体がStrategyGene型ではありません: {type(ind)}")
                    continue

                best_strategies.append(
                    {"strategy": gene, "fitness_values": list(ind.fitness.values)}  # type: ignore[union-attr]
                )
        else:
            # 単一目的最適化の場合
            two_stage_best = get_two_stage_best_individual(population)
            if two_stage_best is not None:
                best_individual = two_stage_best
            elif halloffame is not None and len(halloffame) > 0:
                best_individual = halloffame[0]
            else:
                best_individual = tools.selBest(population, 1)[0]

        if isinstance(best_individual, StrategyGene):
            best_gene = best_individual
        else:
            logger.error(
                f"最良個体がStrategyGene型ではありません: {type(best_individual)}"
            )
            best_gene = None

        return best_individual, best_gene, best_strategies

    def rank_population_for_persistence(self, population: List[Any]) -> List[Any]:
        """
        保存順序用に個体群を安定ソートする。
        """

        def sort_key(individual: Any) -> Tuple[int, int, float]:
            """
            ソート用のキーを生成する。

            2段階評価のランクがあればそれを最優先し、なければ後回しにする。
            同じランク内では、プライマリフィットネスの降順でソートする。
            """
            rank = get_two_stage_rank(individual)
            if rank is not None:
                return (0, rank, -extract_primary_fitness(individual))
            return (1, 0, -extract_primary_fitness(individual))

        return sorted(population, key=sort_key)

    def get_strategy_result_key(self, strategy: Any) -> str:
        """
        result 内部で戦略 summary を対応付けるキーを返す。
        """
        strategy_id = getattr(strategy, "id", None)
        if strategy_id not in (None, ""):
            return str(strategy_id)
        return str(id(strategy))
