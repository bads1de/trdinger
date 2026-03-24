"""
遺伝的アルゴリズムエンジン

DEAPライブラリを使用したGA実装。
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from deap import tools

from app.services.backtest.backtest_service import BacktestService

from ..config.ga import GAConfig
from ..generators.random_gene_generator import RandomGeneGenerator
from ..genes import StrategyGene
from .deap_setup import DEAPSetup
from .evolution_runner import EvolutionRunner
from .fitness_sharing import FitnessSharing
from .ga_utils import (
    _gene_kwargs,
    create_deap_mutate_wrapper,
    crossover_strategy_genes,
    mutate_strategy_gene,
)
from .individual_evaluator import IndividualEvaluator
from .parallel_evaluator import ParallelEvaluator

logger = logging.getLogger(__name__)


class GeneticAlgorithmEngine:
    """遺伝的アルゴリズムエンジン。

    DEAPライブラリを使用して戦略の自動生成・最適化を行います。
    複雑な分離構造を削除し、直接的で理解しやすい実装に変更しました。
    """

    def __init__(
        self,
        backtest_service: BacktestService,
        gene_generator: RandomGeneGenerator,
        hybrid_mode: bool = False,
        hybrid_predictor: Optional[Any] = None,
        hybrid_feature_adapter: Optional[Any] = None,
    ):
        """初期化します。

        Args:
            backtest_service (BacktestService): バックテストサービス。
            gene_generator (RandomGeneGenerator): 遺伝子生成器。
            hybrid_mode (bool): ハイブリッドGA+MLモードを有効化。デフォルトはFalse。
            hybrid_predictor (Optional[Any]): ハイブリッド予測器（hybrid_mode=Trueの場合）。
            hybrid_feature_adapter (Optional[Any]): 特徴量アダプタ（hybrid_mode=Trueの場合）。
        """
        self.backtest_service = backtest_service
        self.gene_generator = gene_generator
        self.hybrid_mode = hybrid_mode

        # 実行状態
        self.is_running = False

        # 分離されたコンポーネント
        self.deap_setup = DEAPSetup()

        # ハイブリッドモードに応じてEvaluatorを選択
        if hybrid_mode:
            logger.info("[Hybrid] ハイブリッドGA+MLモードで起動")
            from .hybrid_individual_evaluator import HybridIndividualEvaluator

            self.individual_evaluator = HybridIndividualEvaluator(
                backtest_service=backtest_service,
                predictor=hybrid_predictor,
                feature_adapter=hybrid_feature_adapter,
            )
        else:
            logger.info("[Standard] 標準GAモードで起動")
            self.individual_evaluator = IndividualEvaluator(backtest_service)

        self.individual_class = None  # setup_deap時に設定
        self.fitness_sharing = None  # setup_deap時に初期化

    def setup_deap(self, config: GAConfig) -> None:
        """
        DEAP フレームワークのコア設定（個体定義、演算子登録）を実行

        `creator.create` を用いて、適応度（高ければ高いほど良い）と
        個体クラス（`StrategyGene` を継承）を動的に定義します。
        その後、選択、交叉、突然変異の各演算子をツールボックスに登録します。

        Args:
            config: 世代数や個体数、報酬設計等の GA 設定
        """
        # 単一目的 or 多目的の設定
        # DEAP環境をセットアップ（戦略個体生成メソッドで統合）
        self.deap_setup.setup_deap(
            config,
            self._create_strategy_individual,
            self.individual_evaluator.evaluate,
            crossover_strategy_genes,
            mutate_strategy_gene,
        )

        # 個体クラスを取得（個体生成時に使用）
        self.individual_class = self.deap_setup.get_individual_class()

        # フィットネス共有の初期化
        if config.enable_fitness_sharing:
            self.fitness_sharing = FitnessSharing(
                sharing_radius=config.sharing_radius,
                alpha=config.sharing_alpha,
                sampling_threshold=config.sampling_threshold,
                sampling_ratio=config.sampling_ratio,
            )
        else:
            self.fitness_sharing = None

        logger.info("DEAP環境のセットアップ完了")

    def run_evolution(
        self, config: GAConfig, backtest_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        進化計算プロセスを開始し、最適な取引戦略を探索

        設定に基づき、初期集団の生成から、評価・選択・交叉・突然変異の
        繰り返し（世代交代）を行い、最終的な最良個体群を抽出します。
        多目的最適化（NSGA-II 等）と単一目的最適化の両方をサポートします。

        Args:
            config: 最適化のアルゴリズムパラメータ（世代数、突然変異率等）
            backtest_config: 個体評価に使用するバックテストの設定（銘柄、期間等）

        Returns:
            最良戦略の遺伝子、評価ログ、実行統計等を含む結果レポート
        """
        try:
            self.is_running = True
            start_time = time.time()

            logger.info(
                "GA Engine - Starting evolution with backtest_config: %s",
                backtest_config,
            )

            # バックテスト設定にデフォルトの日付を設定（存在しない場合）
            if "start_date" not in backtest_config:
                backtest_config["start_date"] = config.fallback_start_date
                logger.info(
                    "GA Engine - Using fallback start_date: %s",
                    config.fallback_start_date,
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

            # 初期個体群の生成（評価なし）
            population = toolbox.population(n=config.population_size)

            # シード戦略の注入（ハイブリッド初期化）
            if config.use_seed_strategies:
                from ..generators.seed_strategy_factory import (
                    SeedStrategyFactory,
                )

                seeds = SeedStrategyFactory.get_all_seeds()
                num_to_inject = min(
                    int(len(population) * config.seed_injection_rate),
                    len(seeds),
                )
                if num_to_inject > 0:
                    individual_class = self.deap_setup.get_individual_class()
                    for i in range(num_to_inject):
                        seed = seeds[i % len(seeds)]
                        population[i] = individual_class(**_gene_kwargs(seed))
                    logger.info(
                        f"シード戦略を {num_to_inject} 個注入しました "
                        f"(注入率: {config.seed_injection_rate * 100:.1f}%)"
                    )

            # 適応的突然変異用mutate_wrapperの設定
            individual_class = self.deap_setup.get_individual_class()
            mutate_wrapper = create_deap_mutate_wrapper(
                individual_class, population, config
            )
            toolbox.register("mutate", mutate_wrapper)

            # 独立したEvolutionRunnerの作成（並列評価対応）
            parallel_evaluator = self._create_parallel_evaluator(config)

            # 並列評価器の起動
            if parallel_evaluator:
                parallel_evaluator.start()

            try:
                runner = self._create_evolution_runner(
                    toolbox, stats, population, config, parallel_evaluator
                )

                # 初期個体群の評価（並列評価対応）
                runner._evaluate_population(population)

                # 最適化アルゴリズムの実行
                population, logbook, halloffame = self._run_optimization(
                    runner, population, config
                )

                # 最良個体の処理と結果生成
                result = self._process_results(
                    population, config, logbook, start_time, halloffame
                )

                logger.info(f"進化完了 - 実行時間: {result['execution_time']:.2f}秒")
                return result

            finally:
                # 並列評価器の停止（確実にリソース解放）
                if parallel_evaluator:
                    parallel_evaluator.shutdown()

        except Exception as e:
            logger.error(f"進化実行エラー: {e}")
            raise
        finally:
            self.is_running = False

    def _set_generator_context(self, backtest_config: Dict[str, Any]):
        """ジェネレーターにコンテキストを設定します。

        Args:
            backtest_config (Dict[str, Any]): バックテスト設定。
        """
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
        """
        統計情報収集オブジェクトを作成

        世代ごとのフィットネスの平均、標準偏差、最小値、最大値を
        記録するためのDEAP Statisticsオブジェクトを構築します。

        Returns:
            統計情報収集用オブジェクト
        """
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        return stats

    def _create_parallel_evaluator(self, config: GAConfig):
        """並列評価器を作成します。

        Args:
            config (GAConfig): GA設定。

        Returns:
            ParallelEvaluator: 並列評価器インスタンス、またはNone。
        """
        if not getattr(config, "enable_parallel_evaluation", False):
            return None

        from .evaluation_worker import (
            initialize_worker_process,
            worker_evaluate_individual,
        )

        # 並列ワーカー用のデータ準備
        worker_initargs = ()

        try:
            # バックテスト設定の取得
            backtest_config = getattr(
                self.individual_evaluator, "_fixed_backtest_config", {}
            )
            if not backtest_config:
                logger.warning(
                    "バックテスト設定が見つかりません。並列評価をスキップします。"
                )
                return None

            shared_data = {}
            # メインデータを取得（キャッシュになければロードされる）
            main_data = self.individual_evaluator._get_cached_data(backtest_config)
            if main_data is not None and not main_data.empty:
                shared_data["main_data"] = main_data

            # 1分足データを取得（存在する場合）
            minute_data = self.individual_evaluator._get_cached_minute_data(
                backtest_config
            )
            if minute_data is not None:
                shared_data["minute_data"] = minute_data

            # 初期化引数: (backtest_config, ga_config, shared_data)
            worker_initargs = (backtest_config, config, shared_data)

            logger.info("並列ワーカー用の初期化パラメータを準備しました")

        except Exception as e:
            logger.warning(f"並列ワーカー用データ準備中にエラーが発生しました: {e}")
            return None

        parallel_evaluator = ParallelEvaluator(
            evaluate_func=worker_evaluate_individual,  # トップレベル関数を指定
            max_workers=getattr(config, "max_evaluation_workers", None),
            timeout_per_individual=getattr(config, "evaluation_timeout", 300.0),
            worker_initializer=initialize_worker_process,  # トップレベル関数を指定
            worker_initargs=worker_initargs,
            use_process_pool=True,
        )
        logger.info(
            f"[Parallel] 並列評価有効: max_workers={parallel_evaluator.max_workers}"
        )
        return parallel_evaluator

    def _create_evolution_runner(
        self,
        toolbox,
        stats,
        population=None,
        config=None,
        parallel_evaluator=None,
    ):
        """独立したEvolutionRunnerインスタンスを作成します。

        Args:
            toolbox: DEAPツールボックス。
            stats: 統計情報オブジェクト。
            population: 初期個体群（オプション）。
            config: GA設定（並列評価用）。
            parallel_evaluator: 事前に作成された並列評価器。

        Returns:
            EvolutionRunner: EvolutionRunnerインスタンス。
        """
        fitness_sharing = (
            self.fitness_sharing
            if hasattr(self, "fitness_sharing") and self.fitness_sharing
            else None
        )

        return EvolutionRunner(
            toolbox, stats, fitness_sharing, population, parallel_evaluator
        )

    def _run_optimization(self, runner: EvolutionRunner, population, config: GAConfig):
        """独立したEvolutionRunnerを使用して最適化アルゴリズムを実行します。

        Args:
            runner (EvolutionRunner): EvolutionRunnerインスタンス。
            population: 初期個体群。
            config (GAConfig): GA設定。

        Returns:
            tuple: 最適化後の個体群、ログブック、殿堂入りオブジェクト。
        """
        # 目的数に応じて適切なHallOfFameオブジェクトを作成
        if config.enable_multi_objective:
            halloffame = tools.ParetoFront()
        else:
            halloffame = tools.HallOfFame(maxsize=1)

        # 統一された進化メソッドを実行
        population, logbook = runner.run_evolution(population, config, halloffame)

        return population, logbook, halloffame

    def _process_results(
        self,
        population,
        config: GAConfig,
        logbook,
        start_time: float,
        halloffame=None,
    ):
        """最適化結果を処理します。

        Args:
            population: 最終個体群。
            config (GAConfig): GA設定。
            logbook: 進化ログ。
            start_time (float): 開始時刻（秒）。

        Returns:
            Dict[str, Any]: 処理された進化結果の辞書。
        """
        # 最良個体の取得とデコード
        best_individual, best_gene, best_strategies = self._extract_best_individuals(
            population, config, halloffame
        )

        # パラメータチューニング（有効な場合）
        if config.enable_parameter_tuning:
            best_gene = self._tune_elite_parameters(best_gene, config)

        execution_time = time.time() - start_time

        # 最終的な結果の構築
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

    def _extract_best_individuals(
        self, population: List[Any], config: GAConfig, halloffame: Optional[Any] = None
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
        best_gene = None  # Initialize best_gene

        if config.enable_multi_objective:
            # 多目的最適化の場合、パレート最適解を取得
            # halloffameがParetoFrontでない場合（fallback）はpopulationから再構築
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
                    # 個体がオブジェクトでない場合は、シリアライザーを使用せずにエラーログを出力
                    logger.error(f"個体がStrategyGene型ではありません: {type(ind)}")
                    continue

                best_strategies.append(
                    {"strategy": gene, "fitness_values": list(ind.fitness.values)}
                )
        else:
            # 単一目的最適化の場合
            if halloffame is not None and len(halloffame) > 0:
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

    def stop_evolution(self):
        """進化を停止します。"""
        self.is_running = False

    def _create_strategy_individual(self):
        """戦略個体生成を行います。

        Returns:
            Individual: Individualオブジェクト。
        """
        try:
            # RandomGeneGeneratorを使用して遺伝子を生成
            gene = self.gene_generator.generate_random_gene()

            if not self.individual_class:
                raise TypeError("個体クラス 'Individual' が初期化されていません。")

            # StrategyGeneのフィールドを使ってIndividualインスタンスを作成
            # IndividualはStrategyGeneを継承しているため、キーワード引数で初期化可能
            # asdictは再帰的に辞書化してしまうため使用しない
            return self.individual_class(**_gene_kwargs(gene))

        except Exception as e:
            logger.error(f"個体生成中に致命的なエラーが発生しました: {e}")
            # 遺伝子生成はGAの根幹部分であり、失敗した場合は例外をスローして処理を停止するのが安全
            raise

    def _tune_elite_parameters(self, best_gene, config: GAConfig):
        """エリート個体のパラメータをOptunaでチューニングします。

        Args:
            best_gene: 最良戦略遺伝子
            config (GAConfig): GA設定

        Returns:
            チューニングされた戦略遺伝子
        """
        try:
            from ..optimization import StrategyParameterTuner

            logger.info("[Tuning] エリート個体のパラメータチューニングを開始")

            tuner = StrategyParameterTuner(
                evaluator=self.individual_evaluator,
                config=config,
                n_trials=config.tuning_n_trials,
                use_wfa=config.tuning_use_wfa,
                include_indicators=config.tuning_include_indicators,
                include_tpsl=config.tuning_include_tpsl,
                include_thresholds=config.tuning_include_thresholds,
            )

            tuned_gene = tuner.tune(best_gene)

            logger.info("[Done] パラメータチューニング完了")
            return tuned_gene

        except Exception as e:
            logger.warning(f"パラメータチューニング中にエラーが発生: {e}")
            # エラー時は元の遺伝子を返す
            return best_gene
