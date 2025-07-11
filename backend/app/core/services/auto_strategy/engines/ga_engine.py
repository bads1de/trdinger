"""
遺伝的アルゴリズムエンジン

DEAPライブラリを使用したGA実装。
"""

import time
import logging
import random
import numpy as np
from typing import Dict, Any
from deap import tools, algorithms

from ..models.ga_config import GAConfig
from ..factories.strategy_factory import StrategyFactory
from ..generators.random_gene_generator import RandomGeneGenerator
from app.core.services.backtest_service import BacktestService
from .deap_setup import DEAPSetup
from .individual_evaluator import IndividualEvaluator
from .evolution_operators import EvolutionOperators
from .individual_creator import IndividualCreator
from .fitness_sharing import FitnessSharing

logger = logging.getLogger(__name__)


class GeneticAlgorithmEngine:
    """
    遺伝的アルゴリズムエンジン

    DEAPライブラリを使用して戦略の自動生成・最適化を行います。
    複雑な分離構造を削除し、直接的で理解しやすい実装に変更しました。
    """

    def __init__(
        self,
        backtest_service: BacktestService,
        strategy_factory: StrategyFactory,
        gene_generator: RandomGeneGenerator,
    ):
        """
        初期化

        Args:
            backtest_service: バックテストサービス
            strategy_factory: 戦略ファクトリー
            gene_generator: 遺伝子生成器
        """
        self.backtest_service = backtest_service
        self.strategy_factory = strategy_factory
        self.gene_generator = gene_generator

        # 実行状態
        self.is_running = False

        # 分離されたコンポーネント
        self.deap_setup = DEAPSetup()
        self.individual_evaluator = IndividualEvaluator(backtest_service)
        self.evolution_operators = EvolutionOperators()
        self.individual_creator = None  # setup_deap時に初期化
        self.fitness_sharing = None  # setup_deap時に初期化

    def setup_deap(self, config: GAConfig):
        """
        DEAP環境のセットアップ（統合版）

        Args:
            config: GA設定
        """
        # 個体生成器を初期化
        self.individual_creator = IndividualCreator(
            self.gene_generator, None  # 個体クラスはDEAPSetupで設定される
        )

        # DEAP環境をセットアップ
        self.deap_setup.setup_deap(
            config,
            self.individual_creator.create_individual,
            self.individual_evaluator.evaluate_individual,
            self.evolution_operators.crossover_strategy_genes,
            self.evolution_operators.mutate_strategy_gene,
        )

        # 個体生成器に個体クラスを設定
        self.individual_creator.Individual = self.deap_setup.get_individual_class()

        # フィットネス共有の初期化
        if config.enable_fitness_sharing:
            self.fitness_sharing = FitnessSharing(
                sharing_radius=config.sharing_radius,
                alpha=config.sharing_alpha
            )
        else:
            self.fitness_sharing = None

        logger.info("DEAP環境のセットアップ完了")

    def run_evolution(
        self, config: GAConfig, backtest_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        進化アルゴリズムを実行

        Args:
            config: GA設定
            backtest_config: バックテスト設定

        Returns:
            進化結果
        """
        try:
            self.is_running = True
            start_time = time.time()

            # バックテスト設定を保存
            self.individual_evaluator.set_backtest_config(backtest_config)

            # DEAP環境のセットアップ
            self.setup_deap(config)

            # ツールボックスを取得
            toolbox = self.deap_setup.get_toolbox()
            assert toolbox is not None, "Toolbox must be initialized before use."

            # 初期個体群の生成
            population = toolbox.population(n=config.population_size)  # type: ignore

            # 統計情報の設定
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)

            # 進化アルゴリズムの実行
            if config.enable_fitness_sharing and self.fitness_sharing:
                # フィットネス共有付き進化
                population, logbook = self._run_evolution_with_fitness_sharing(
                    population,
                    toolbox,
                    config,
                    stats
                )
            else:
                # 通常の進化アルゴリズム
                population, logbook = algorithms.eaSimple(
                    population,
                    toolbox,
                    cxpb=config.crossover_rate,
                    mutpb=config.mutation_rate,
                    ngen=config.generations,
                    stats=stats,
                    verbose=False,
                )

            # 最良個体の取得
            best_individual = tools.selBest(population, 1)[0]

            # 遺伝子デコード
            from ..models.gene_encoding import GeneEncoder
            from ..models.gene_strategy import StrategyGene

            gene_encoder = GeneEncoder()
            best_gene = gene_encoder.decode_list_to_strategy_gene(
                best_individual, StrategyGene
            )

            execution_time = time.time() - start_time

            result = {
                "best_strategy": best_gene,
                "best_fitness": best_individual.fitness.values[0],
                "population": population,
                "logbook": logbook,
                "execution_time": execution_time,
                "generations_completed": config.generations,
                "final_population_size": len(population),
            }

            logger.info(f"進化完了 - 実行時間: {execution_time:.2f}秒")
            return result

        except Exception as e:
            logger.error(f"進化実行エラー: {e}")
            raise
        finally:
            self.is_running = False

    def stop_evolution(self):
        """進化を停止"""
        self.is_running = False

    def _run_evolution_with_fitness_sharing(
        self, population, toolbox, config: GAConfig, stats
    ):
        """
        フィットネス共有を適用した進化アルゴリズムの実行

        Args:
            population: 初期個体群
            toolbox: DEAPツールボックス
            config: GA設定
            stats: 統計情報

        Returns:
            進化後の個体群とログブック
        """
        try:
            logbook = tools.Logbook()
            logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

            # 初期個体群の評価
            fitnesses = toolbox.map(toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # フィットネス共有を適用
            population = self.fitness_sharing.apply_fitness_sharing(population)

            if stats:
                record = stats.compile(population)
                logbook.record(gen=0, nevals=len(population), **record)

            # 世代ループ
            for gen in range(1, config.generations + 1):
                # 選択
                offspring = toolbox.select(population, len(population))
                offspring = [toolbox.clone(ind) for ind in offspring]

                # 交叉と突然変異
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < config.crossover_rate:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                for mutant in offspring:
                    if random.random() < config.mutation_rate:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values

                # 評価が必要な個体を特定
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # フィットネス共有を適用
                offspring = self.fitness_sharing.apply_fitness_sharing(offspring)

                # 次世代の選択
                population[:] = offspring

                # 統計情報の記録
                if stats:
                    record = stats.compile(population)
                    logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            return population, logbook

        except Exception as e:
            logger.error(f"フィットネス共有付き進化実行エラー: {e}")
            # フォールバック: 通常の進化アルゴリズム
            return algorithms.eaSimple(
                population,
                toolbox,
                cxpb=config.crossover_rate,
                mutpb=config.mutation_rate,
                ngen=config.generations,
                stats=stats,
                verbose=False,
            )
