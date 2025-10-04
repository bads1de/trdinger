"""
遺伝的アルゴリズムエンジン

DEAPライブラリを使用したGA実装。
"""

import logging
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
from deap import tools

from app.services.backtest.backtest_service import BacktestService

from ..config import GAConfigPydantic as GAConfig
from ..generators.random_gene_generator import RandomGeneGenerator
from ..generators.strategy_factory import StrategyFactory
from ..services.regime_detector import RegimeDetector
from .deap_setup import DEAPSetup
from .fitness_sharing import FitnessSharing
from .genetic_operators import (
    create_deap_mutate_wrapper,
    crossover_strategy_genes,
    mutate_strategy_gene,
)
from .individual_evaluator import IndividualEvaluator

logger = logging.getLogger(__name__)


class EvolutionRunner:
    """
    進化計算の実行を担当するクラス

    単一目的と多目的最適化のロジックをカプセル化したヘルパークラスです。
    """

    def __init__(self, toolbox, stats, fitness_sharing=None, population=None):
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
        self, population: List[Any], config: GAConfig, halloffame: List[Any] = None
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

        logbook = tools.Logbook()

        # カスタム世代ループ（fitness_sharingを世代毎に適用）
        for gen in range(config.generations):
            logger.debug(f"世代 {gen + 1}/{config.generations} を開始")

            # 適応度共有の適用（有効な場合、世代毎）
            if config.enable_fitness_sharing and self.fitness_sharing:
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

            # 統計の記録
            record = self.stats.compile(population) if self.stats else {}
            logbook.record(gen=gen, **record)

            # Hall of Fameの更新
            if halloffame is not None:
                halloffame.update(population)

        logger.info("単一目的最適化アルゴリズム完了")
        return population, logbook

    def run_multi_objective_evolution(
        self, population: List[Any], config: GAConfig, halloffame: List[Any] = None
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
            if config.enable_fitness_sharing and self.fitness_sharing:
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
        regime_detector: Optional["RegimeDetector"] = None,
        hybrid_mode: bool = False,
        hybrid_predictor: Optional[Any] = None,
        hybrid_feature_adapter: Optional[Any] = None,
    ):
        """
        初期化

        Args:
            backtest_service: バックテストサービス
            strategy_factory: 戦略ファクトリー
            gene_generator: 遺伝子生成器
            regime_detector: レジーム検知器（オプション、レジーム適応時に使用）
            hybrid_mode: ハイブリッドGA+MLモードを有効化
            hybrid_predictor: ハイブリッド予測器（hybrid_mode=Trueの場合）
            hybrid_feature_adapter: 特徴量アダプタ（hybrid_mode=Trueの場合）
        """
        self.backtest_service = backtest_service
        self.strategy_factory = strategy_factory
        self.gene_generator = gene_generator
        self.hybrid_mode = hybrid_mode

        # 実行状態
        self.is_running = False

        # 分離されたコンポーネント
        self.deap_setup = DEAPSetup()
        
        # ハイブリッドモードに応じてEvaluatorを選択
        if hybrid_mode:
            logger.info("🔬 ハイブリッドGA+MLモードで起動")
            from .hybrid_individual_evaluator import HybridIndividualEvaluator
            self.individual_evaluator = HybridIndividualEvaluator(
                backtest_service=backtest_service,
                predictor=hybrid_predictor,
                feature_adapter=hybrid_feature_adapter,
                regime_detector=regime_detector,
            )
        else:
            logger.info("🧬 標準GAモードで起動")
            self.individual_evaluator = IndividualEvaluator(
                backtest_service, regime_detector
            )
        
        self.individual_class = None  # setup_deap時に設定
        self.fitness_sharing = None  # setup_deap時に初期化

    def setup_deap(self, config: GAConfig):
        """
        DEAP環境のセットアップ（統合版）

        Args:
            config: GA設定
        """
        # DEAP環境をセットアップ（個体生成メソッドで統合）
        self.deap_setup.setup_deap(
            config,
            self._create_individual,
            self.individual_evaluator.evaluate_individual,
            crossover_strategy_genes,
            mutate_strategy_gene,
        )

        # 個体クラスを取得（個体生成時に使用）
        self.individual_class = self.deap_setup.get_individual_class()

        # フィットネス共有の初期化
        if config.enable_fitness_sharing:
            self.fitness_sharing = FitnessSharing(
                sharing_radius=config.sharing_radius, alpha=config.sharing_alpha
            )
        else:
            self.fitness_sharing = None

        logger.info("DEAP環境のセットアップ完了")

    def run_evolution(
        self, config: GAConfig, backtest_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        進化アルゴリズムを実行

        EvolutionRunnerを使って設定に応じて適切な最適化アルゴリズムを呼び出します。

        Args:
            config: GA設定
            backtest_config: バックテスト設定

        Returns:
            進化結果
        """
        try:
            self.is_running = True
            start_time = time.time()

            logger.info(
                "GA Engine - Starting evolution with backtest_config: %s",
                backtest_config
            )

            # バックテスト設定にデフォルトの日付を設定（存在しない場合）
            if "start_date" not in backtest_config:
                backtest_config["start_date"] = config.fallback_start_date
                logger.info(
                    "GA Engine - Using fallback start_date: %s",
                    config.fallback_start_date
                )
            if "end_date" not in backtest_config:
                backtest_config["end_date"] = config.fallback_end_date
                logger.info(
                    f"GA Engine - Using fallback end_date: {config.fallback_end_date}"
                )

            # バックテスト設定を保存
            self.individual_evaluator.set_backtest_config(backtest_config)

            # コンテキスト設定（省略可能）
            self._set_generator_context(backtest_config)

            # DEAP環境のセットアップ
            self.setup_deap(config)

            # ツールボックスと統計情報の取得
            toolbox = self.deap_setup.get_toolbox()
            assert toolbox is not None, "Toolbox must be initialized before use."

            stats = self._create_statistics()

            # 初期個体群の生成と評価
            population = self._create_initial_population(toolbox, config)

            # 適応的突然変異用mutate_wrapperの設定
            individual_class = self.deap_setup.get_individual_class()
            mutate_wrapper = create_deap_mutate_wrapper(individual_class, population)
            toolbox.register("mutate", mutate_wrapper)

            # EvolutionRunnerの作成
            runner = self._create_evolution_runner(toolbox, stats, population)

            # 最適化アルゴリズムの実行
            population, logbook = self._run_optimization(runner, population, config)

            # 最良個体の処理と結果生成
            result = self._process_results(population, config, logbook, start_time)

            logger.info(f"進化完了 - 実行時間: {result['execution_time']:.2f}秒")
            return result

        except Exception as e:
            logger.error(f"進化実行エラー: {e}")
            raise
        finally:
            self.is_running = False

    def _set_generator_context(self, backtest_config: Dict[str, Any]):
        """ジェネレーターにコンテキストを設定"""
        try:
            tf = backtest_config.get("timeframe")
            sym = backtest_config.get("symbol")
            if hasattr(self.gene_generator, "smart_condition_generator"):
                smart_gen = getattr(self.gene_generator, "smart_condition_generator")
                if smart_gen and hasattr(smart_gen, "set_context"):
                    smart_gen.set_context(timeframe=tf, symbol=sym)
        except Exception:
            pass

    def _create_statistics(self):
        """統計情報収集オブジェクトを作成"""
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        return stats

    def _create_evolution_runner(self, toolbox, stats, population=None):
        """EvolutionRunnerインスタンスを作成"""
        fitness_sharing = (
            self.fitness_sharing
            if hasattr(self, "fitness_sharing") and self.fitness_sharing
            else None
        )
        return EvolutionRunner(toolbox, stats, fitness_sharing, population)

    def _create_initial_population(self, toolbox, config: GAConfig):
        """初期個体群を生成"""
        population = toolbox.population(n=config.population_size)
        # 初期評価
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        return population

    def _run_optimization(self, runner: EvolutionRunner, population, config: GAConfig):
        """最適化アルゴリズムを実行"""
        if config.enable_multi_objective:
            return runner.run_multi_objective_evolution(population, config)
        else:
            return runner.run_single_objective_evolution(population, config)

    def _process_results(
        self, population, config: GAConfig, logbook, start_time: float
    ):
        """最適化結果を処理"""
        # 最良個体の取得とデコード
        best_individual, best_gene, best_strategies = self._extract_best_individuals(
            population, config
        )

        execution_time = time.time() - start_time

        result = {
            "best_strategy": best_gene,
            "best_fitness": (
                best_individual.fitness.values[0]
                if not config.enable_multi_objective
                else best_individual.fitness.values
            ),
            "population": population,
            "logbook": logbook,
            "execution_time": execution_time,
            "generations_completed": config.generations,
            "final_population_size": len(population),
        }

        # 多目的最適化の場合、パレート最適解を追加
        if config.enable_multi_objective:
            result["pareto_front"] = best_strategies
            result["objectives"] = config.objectives

        return result

    def _extract_best_individuals(self, population, config: GAConfig):
        """最良個体を抽出し、デコード"""
        from ..models.strategy_models import StrategyGene
        from ..serializers.gene_serialization import GeneSerializer

        gene_serializer = GeneSerializer()

        if config.enable_multi_objective:
            # 多目的最適化の場合、パレート最適解を取得
            pareto_front = tools.ParetoFront()
            pareto_front.update(population)
            best_individuals = list(pareto_front)
            best_individual = best_individuals[0] if best_individuals else population[0]

            best_strategies = []
            for ind in best_individuals[:10]:  # 上位10個のパレート最適解
                gene = gene_serializer.from_list(ind, StrategyGene)
                best_strategies.append(
                    {"strategy": gene, "fitness_values": list(ind.fitness.values)}
                )

            best_gene = gene_serializer.from_list(best_individual, StrategyGene)
        else:
            # 単一目的最適化の場合
            best_individual = tools.selBest(population, 1)[0]
            best_gene = gene_serializer.from_list(best_individual, StrategyGene)
            best_strategies = None

        return best_individual, best_gene, best_strategies

    def stop_evolution(self):
        """進化を停止"""
        self.is_running = False

    def _create_individual(self):
        """
        個体生成（統合版IndividualCreator）

        Returns:
            Individualオブジェクト
        """
        try:
            # RandomGeneGeneratorを使用して遺伝子を生成
            gene = self.gene_generator.generate_random_gene()

            # 遺伝子をエンコード（リファクタリング改善）
            from ..serializers.gene_serialization import GeneSerializer

            gene_serializer = GeneSerializer()
            encoded_gene = gene_serializer.to_list(gene)

            if not self.individual_class:
                raise TypeError("個体クラス 'Individual' が初期化されていません。")
            return self.individual_class(encoded_gene)

        except Exception as e:
            logger.error(f"個体生成中に致命的なエラーが発生しました: {e}")
            # 遺伝子生成はGAの根幹部分であり、失敗した場合は例外をスローして処理を停止するのが安全
            raise
