"""
遺伝的アルゴリズムエンジン

DEAPライブラリを使用したGA実装。
"""

import logging
import threading
import time
from math import isfinite
from typing import Any, Dict, List, Optional, Tuple, cast


from ..evaluation.evaluation_report import EvaluationReport
from ..evaluation.evaluation_fidelity import (
    build_coarse_ga_config,
    is_multi_fidelity_enabled,
)

import numpy as np
from deap import tools

from app.services.auto_strategy.config.ga import GAConfig
from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.services.auto_strategy.genes import StrategyGene
from app.services.backtest.services.backtest_service import BacktestService

from ..evaluation.individual_evaluator import IndividualEvaluator
from ..evaluation.parallel_evaluator import ParallelEvaluator
from ..evaluation.report_persistence import build_report_summary
from ..fitness.fitness_sharing import FitnessSharing
from .deap_setup import DEAPSetup
from .evolution_runner import EvolutionRunner, EvolutionStoppedError
from .ga_utils import (
    _gene_kwargs,
    create_deap_mutate_wrapper,
    crossover_strategy_genes,
    mutate_strategy_gene,
)
from .fitness_utils import (
    extract_primary_fitness_from_result,
    extract_result_fitness,
)
from .report_selection import (
    build_report_rank_key_from_primary_fitness,
    extract_primary_fitness,
    get_two_stage_best_individual,
    get_two_stage_rank,
    get_two_stage_score,
    is_evaluation_report,
)

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
        self._stop_event = threading.Event()

        # 分離されたコンポーネント
        self.deap_setup = DEAPSetup()

        # ハイブリッドモードに応じてEvaluatorを選択
        if hybrid_mode:
            logger.info("[Hybrid] ハイブリッドGA+MLモードで起動")
            from ..hybrid.hybrid_individual_evaluator import (
                HybridIndividualEvaluator,
            )

            self.individual_evaluator = HybridIndividualEvaluator(
                backtest_service=backtest_service,
                predictor=hybrid_predictor,
                feature_adapter=hybrid_feature_adapter,
            )
        else:
            logger.info("[Standard] 標準GAモードで起動")
            self.individual_evaluator = IndividualEvaluator(backtest_service)

        self.individual_class = None  # setup_deap時に設定
        self.fitness_sharing: Any = None  # setup_deap時に初期化

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
        fitness_sharing_config = config.fitness_sharing
        if fitness_sharing_config.get("enable_fitness_sharing", False):
            self.fitness_sharing = FitnessSharing(
                sharing_radius=fitness_sharing_config.get("sharing_radius", 0.1),
                alpha=fitness_sharing_config.get("sharing_alpha", 1.0),
                sampling_threshold=fitness_sharing_config.get("sampling_threshold", 200),
                sampling_ratio=fitness_sharing_config.get(
                    "sampling_ratio", FitnessSharing.SAMPLING_RATIO
                ),
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

            self._raise_if_stop_requested("開始前")

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

            self._raise_if_stop_requested("バックテスト設定準備後")

            # バックテスト設定を保存
            self.individual_evaluator.set_backtest_config(backtest_config)

            # コンテキスト設定（省略可能）
            self._set_generator_context(backtest_config)

            self._raise_if_stop_requested("コンテキスト設定後")

            # DEAP環境のセットアップ
            self.setup_deap(config)

            self._raise_if_stop_requested("DEAPセットアップ後")

            # ツールボックスと統計情報の取得
            toolbox = self.deap_setup.get_toolbox()
            if toolbox is None:
                raise RuntimeError("Toolbox must be initialized before use.")

            stats = self._create_statistics()

            # 初期個体群の生成（評価なし）
            population = toolbox.population(n=config.population_size)  # type: ignore[attr-defined]

            # シード戦略の注入（ハイブリッド初期化）
            if config.use_seed_strategies:
                from app.services.auto_strategy.generators.seed_strategy_factory import (
                    SeedStrategyFactory,
                )

                seeds = SeedStrategyFactory.get_all_seeds()
                num_to_inject = min(
                    int(len(population) * config.seed_injection_rate),
                    len(seeds),
                )
                if num_to_inject > 0:
                    individual_class = self.deap_setup.get_individual_class()
                    if individual_class is not None:
                        for i in range(num_to_inject):
                            seed = seeds[i % len(seeds)]
                            population[i] = individual_class(**_gene_kwargs(seed))
                    else:
                        logger.warning(
                            "個体クラスが未初期化のため、シード戦略の注入をスキップしました"
                        )
                    logger.info(
                        f"シード戦略を {num_to_inject} 個注入しました "
                        f"(注入率: {config.seed_injection_rate * 100:.1f}%)"
                    )

            self._raise_if_stop_requested("初期集団生成後")

            # 適応的突然変異用mutate_wrapperの設定
            individual_class = self.deap_setup.get_individual_class()
            mutate_wrapper = create_deap_mutate_wrapper(
                individual_class, population, config
            )
            toolbox.register("mutate", mutate_wrapper)

            # 独立したEvolutionRunnerの作成（並列評価対応）
            parallel_evaluator = self._create_parallel_evaluator(config)

            try:
                # 並列評価器の起動
                if parallel_evaluator:
                    parallel_evaluator.start()

                self._raise_if_stop_requested("進化実行前")

                runner = self._create_evolution_runner(
                    toolbox, stats, population, config, parallel_evaluator
                )

                # 最適化アルゴリズムの実行
                population, logbook, halloffame = self._run_optimization(
                    runner, population, config
                )

                self._raise_if_stop_requested("最適化後")

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

        except EvolutionStoppedError:
            logger.info("進化実行が停止されました")
            raise
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
        except (AttributeError, TypeError) as e:
            logger.debug(f"コンテキスト設定スキップ: {e}")

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

        from ..evaluation.evaluation_worker import (
            initialize_worker_process,
            worker_evaluate_individual,
        )

        # 並列ワーカー用のデータ準備
        worker_initargs = ()

        try:
            worker_config = (
                build_coarse_ga_config(config)
                if is_multi_fidelity_enabled(config)
                else config
            )
            worker_initargs = self.individual_evaluator.build_parallel_worker_initargs(
                worker_config
            )
            if not worker_initargs:
                logger.warning(
                    "バックテスト設定が見つかりません。並列評価をスキップします。"
                )
                return None

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
            toolbox,
            stats,
            fitness_sharing,
            population,
            parallel_evaluator,
            self.individual_evaluator,
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
        population, logbook = runner.run_evolution(
            population,
            config,
            halloffame,
            should_stop=self._stop_event.is_set,
        )

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

        if best_individual is not None and is_multi_fidelity_enabled(config):
            try:
                refreshed = self._evaluate_individual_with_full_fidelity(
                    best_individual,
                    config,
                )
                if getattr(best_individual, "fitness", None) is not None:
                    best_individual.fitness.values = tuple(refreshed)
            except Exception as exc:
                logger.warning("最終候補の full 評価に失敗しました: %s", exc)

        best_fitness_value = self._extract_result_best_fitness(best_individual, config)
        best_evaluation_summary = self._build_individual_evaluation_summary(
            best_individual, config
        )

        # パラメータチューニング（有効な場合）
        if config.tuning_config.enabled:
            (
                best_gene,
                best_fitness_value,
                best_evaluation_summary,
            ) = self._tune_and_select_best_gene(
                population=population,
                current_best_gene=best_gene,
                config=config,
                fallback_fitness=best_fitness_value,
                fallback_summary=best_evaluation_summary,
            )
        elif best_evaluation_summary is None:
            best_evaluation_summary = self._build_individual_evaluation_summary(
                best_gene,
                config,
                force_robustness=bool(
                    config.two_stage_selection_config.enabled
                ),
            )

        execution_time = time.time() - start_time

        # 最終的な結果の構築
        result = {
            "best_strategy": best_gene,
            "best_fitness": best_fitness_value,
            "population": population,
            "logbook": logbook,
            "execution_time": execution_time,
            "generations_completed": config.generations,
            "final_population_size": len(population),
            "best_evaluation_summary": best_evaluation_summary,
        }

        if not config.enable_multi_objective:
            ranked_population = self._rank_population_for_persistence(population)
            persisted_population = ranked_population[:100]
            result["all_strategies"] = persisted_population
            result["fitness_scores"] = [
                extract_primary_fitness(individual)
                for individual in persisted_population
            ]
            result["evaluation_summaries"] = (
                self._collect_population_evaluation_summaries(
                    persisted_population,
                    config,
                )
            )

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

    def stop_evolution(self):
        """進化を停止します。"""
        self._stop_event.set()
        self.is_running = False

    def is_stop_requested(self) -> bool:
        """停止要求が出ているかを返します。"""
        return self._stop_event.is_set()

    def _raise_if_stop_requested(self, context: str = "") -> None:
        """停止要求がある場合は EvolutionStoppedError を送出します。"""
        if self._stop_event.is_set():
            if context:
                raise EvolutionStoppedError(f"停止要求により中断されました: {context}")
            raise EvolutionStoppedError("停止要求により中断されました")

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
            from app.services.auto_strategy.optimization import StrategyParameterTuner

            logger.info("[Tuning] エリート個体のパラメータチューニングを開始")

            tuner = StrategyParameterTuner.from_ga_config(
                self.individual_evaluator,
                config,
            )

            tuned_gene = tuner.tune(best_gene)

            logger.info("[Done] パラメータチューニング完了")
            return tuned_gene

        except Exception as e:
            logger.warning(f"パラメータチューニング中にエラーが発生: {e}")
            # エラー時は元の遺伝子を返す
            return best_gene

    def _tune_and_select_best_gene(
        self,
        *,
        population: List[Any],
        current_best_gene: Optional[StrategyGene],
        config: GAConfig,
        fallback_fitness: Any,
        fallback_summary: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[StrategyGene], Any, Optional[Dict[str, Any]]]:
        """上位候補をチューニングし、設定に応じた基準で最終勝者を選び直す。"""
        if current_best_gene is None:
            return current_best_gene, fallback_fitness, fallback_summary

        if config.enable_multi_objective:
            tuned_gene = self._tune_elite_parameters(current_best_gene, config)
            refreshed_fitness, refreshed_summary = self._refresh_best_gene_reporting(
                best_gene=tuned_gene,
                config=config,
                fallback_fitness=fallback_fitness,
                fallback_summary=fallback_summary,
            )
            return tuned_gene, refreshed_fitness, refreshed_summary

        tuning_candidates = self._select_tuning_candidates(
            population,
            config,
            fallback_gene=current_best_gene,
        )
        if not tuning_candidates:
            refreshed_fitness, refreshed_summary = self._refresh_best_gene_reporting(
                best_gene=current_best_gene,
                config=config,
                fallback_fitness=fallback_fitness,
                fallback_summary=fallback_summary,
            )
            return current_best_gene, refreshed_fitness, refreshed_summary

        tuned_candidates = self._tune_candidate_genes(tuning_candidates, config)
        if config.two_stage_selection_config.enabled:
            tuned_winner = self._select_best_tuned_candidate(
                tuned_candidates,
                config,
            )
        else:
            tuned_winner = self._select_best_tuned_candidate_by_fitness(
                tuned_candidates,
                config,
            )
        if tuned_winner is None:
            refreshed_fitness, refreshed_summary = self._refresh_best_gene_reporting(
                best_gene=current_best_gene,
                config=config,
                fallback_fitness=fallback_fitness,
                fallback_summary=fallback_summary,
            )
            return current_best_gene, refreshed_fitness, refreshed_summary

        if config.two_stage_selection_config.enabled:
            logger.info(
                "[Tuning] %s候補をチューニングし、robustness 再選抜で最終勝者を決定しました",
                len(tuned_candidates),
            )
        else:
            logger.info(
                "[Tuning] %s候補をチューニングし、主 fitness で最終勝者を決定しました",
                len(tuned_candidates),
            )
        return tuned_winner

    def _extract_result_best_fitness(
        self, best_individual: Any, config: GAConfig
    ) -> Any:
        """結果出力用の best fitness を抽出する。"""
        return extract_result_fitness(
            best_individual,
            enable_multi_objective=config.enable_multi_objective,
        )

    def _rank_population_for_persistence(self, population: List[Any]) -> List[Any]:
        """保存順序用に個体群を安定ソートする。"""

        def sort_key(individual: Any) -> Tuple[int, int, float]:
            """ソート用のキーを生成する。

            2段階評価のランクがあればそれを最優先し、なければ後回しにする。
            同じランク内では、プライマリフィットネスの降順でソートする。
            """
            rank = get_two_stage_rank(individual)
            if rank is not None:
                return (0, rank, -extract_primary_fitness(individual))
            return (1, 0, -extract_primary_fitness(individual))

        return sorted(population, key=sort_key)

    def _collect_population_evaluation_summaries(
        self,
        population: List[Any],
        config: GAConfig,
    ) -> Dict[str, Dict[str, Any]]:
        """保存対象個体の評価 summary を収集する。"""
        summaries: Dict[str, Dict[str, Any]] = {}
        for individual in population:
            summary = self._build_individual_evaluation_summary(individual, config)
            if not summary:
                continue
            strategy_key = self._get_strategy_result_key(individual)
            summaries[strategy_key] = summary
        return summaries

    def _refresh_best_gene_reporting(
        self,
        *,
        best_gene: Optional[StrategyGene],
        config: GAConfig,
        fallback_fitness: Any,
        fallback_summary: Optional[Dict[str, Any]],
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """チューニング後の最良遺伝子を再評価し、summary を最新化する。"""
        if best_gene is None:
            return fallback_fitness, fallback_summary

        refreshed_fitness = fallback_fitness
        try:
            evaluated = self._evaluate_individual_with_full_fidelity(best_gene, config)
            if config.enable_multi_objective:
                refreshed_fitness = tuple(evaluated)
            elif isinstance(evaluated, (tuple, list)) and evaluated:
                refreshed_fitness = float(evaluated[0])
        except Exception as exc:
            logger.warning("最良遺伝子の再評価に失敗しました: %s", exc)

        refreshed_summary = self._build_individual_evaluation_summary(
            best_gene,
            config,
            force_robustness=bool(config.two_stage_selection_config.enabled),
            primary_fitness=self._extract_primary_fitness_from_result(
                refreshed_fitness
            ),
        )
        return refreshed_fitness, refreshed_summary or fallback_summary

    def _select_tuning_candidates(
        self,
        population: List[Any],
        config: GAConfig,
        *,
        fallback_gene: Optional[StrategyGene] = None,
    ) -> List[StrategyGene]:
        """チューニング対象の上位候補を抽出する。"""
        budget = getattr(config, "tuning_elite_count", 1)
        try:
            candidate_budget = max(1, int(budget))
        except (TypeError, ValueError):
            candidate_budget = 1

        ordered_population = self._rank_population_for_persistence(population)
        candidates: List[StrategyGene] = []
        seen_keys = set()

        for individual in ordered_population:
            if not isinstance(individual, StrategyGene):
                continue
            identity = self._get_strategy_result_key(individual)
            if identity in seen_keys:
                continue
            seen_keys.add(identity)
            candidates.append(individual)
            if len(candidates) >= candidate_budget:
                break

        if fallback_gene is not None and not candidates:
            candidates.append(fallback_gene)

        return candidates

    def _tune_candidate_genes(
        self,
        candidates: List[StrategyGene],
        config: GAConfig,
    ) -> List[StrategyGene]:
        """候補遺伝子群を順次チューニングする。"""
        from app.services.auto_strategy.optimization import StrategyParameterTuner

        tuner = StrategyParameterTuner.from_ga_config(
            self.individual_evaluator,
            config,
        )

        tuned_candidates: List[StrategyGene] = []
        for candidate_rank, candidate in enumerate(candidates):
            try:
                tuned_candidate = tuner.tune(candidate)
            except Exception as exc:
                logger.warning(
                    "[Tuning] 候補 %s のチューニングに失敗: %s", candidate_rank, exc
                )
                tuned_candidate = candidate
            tuned_candidate.metadata.setdefault("tuning_candidate_rank", candidate_rank)
            tuned_candidates.append(tuned_candidate)

        return tuned_candidates

    def _select_best_tuned_candidate(
        self,
        tuned_candidates: List[StrategyGene],
        config: GAConfig,
    ) -> Optional[Tuple[StrategyGene, float, Optional[Dict[str, Any]]]]:
        """チューニング後候補を robustness で再評価し最終勝者を返す。"""
        if not tuned_candidates:
            return None

        best_tuple: Optional[Tuple[StrategyGene, float, Optional[Dict[str, Any]]]] = (
            None
        )
        best_key = None

        for candidate_rank, candidate in enumerate(tuned_candidates):
            try:
                fitness_result = self._evaluate_individual_with_full_fidelity(
                    candidate,
                    config,
                )
            except Exception as exc:
                logger.warning(
                    "[Tuning] 候補 %s の再評価に失敗: %s",
                    candidate_rank,
                    exc,
                )
                continue

            primary_fitness = self._extract_primary_fitness_from_result(fitness_result)
            report = None
            evaluate_robustness_report = getattr(
                self.individual_evaluator,
                "evaluate_robustness_report",
                None,
            )
            if callable(evaluate_robustness_report):
                try:
                    report = evaluate_robustness_report(candidate, config)
                except Exception as exc:
                    logger.debug(
                        "[Tuning] 候補 %s の robustness 評価に失敗: %s",
                        candidate_rank,
                        exc,
                    )

            if not is_evaluation_report(report):
                get_cached_evaluation_report = getattr(
                    self.individual_evaluator,
                    "get_cached_evaluation_report",
                    None,
                )
                if callable(get_cached_evaluation_report):
                    report = get_cached_evaluation_report(candidate)

            rank_key = build_report_rank_key_from_primary_fitness(
                primary_fitness,
                cast(Optional["EvaluationReport"], report if is_evaluation_report(report) else None),  # type: ignore[reportArgumentType]
                min_pass_rate=float(
                    getattr(config, "two_stage_min_pass_rate", 0.0) or 0.0
                ),
            )
            summary = self._build_individual_evaluation_summary(
                candidate,
                config,
                force_robustness=False,
                primary_fitness=primary_fitness,
                selection_rank_override=candidate_rank,
                selection_score_override=rank_key,
            )
            candidate_result = (
                candidate,
                primary_fitness,
                summary,
            )
            if best_key is None or rank_key > best_key:
                best_key = rank_key
                best_tuple = candidate_result

        return best_tuple

    def _select_best_tuned_candidate_by_fitness(
        self,
        tuned_candidates: List[StrategyGene],
        config: GAConfig,
    ) -> Optional[Tuple[StrategyGene, float, Optional[Dict[str, Any]]]]:
        """チューニング後候補を主 fitness だけで再選抜する。"""
        if not tuned_candidates:
            return None

        best_tuple: Optional[Tuple[StrategyGene, float, Optional[Dict[str, Any]]]] = (
            None
        )
        best_fitness: Optional[float] = None

        for candidate in tuned_candidates:
            try:
                fitness_result = self._evaluate_individual_with_full_fidelity(
                    candidate,
                    config,
                )
            except Exception as exc:
                logger.warning("[Tuning] 候補の再評価に失敗: %s", exc)
                continue

            primary_fitness = self._extract_primary_fitness_from_result(fitness_result)
            summary = self._build_individual_evaluation_summary(
                candidate,
                config,
                force_robustness=False,
                primary_fitness=primary_fitness,
            )
            candidate_result = (
                candidate,
                primary_fitness,
                summary,
            )
            if best_fitness is None or primary_fitness > best_fitness:
                best_fitness = primary_fitness
                best_tuple = candidate_result

        return best_tuple

    def _build_individual_evaluation_summary(
        self,
        individual: Any,
        config: GAConfig,
        *,
        force_robustness: bool = False,
        primary_fitness: Optional[float] = None,
        selection_rank_override: Optional[int] = None,
        selection_score_override: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """個体の評価 report から保存向け summary を構築する。"""
        if individual is None:
            return None

        get_cached_robustness_report = getattr(
            self.individual_evaluator,
            "get_cached_robustness_report",
            None,
        )
        evaluate_robustness_report = getattr(
            self.individual_evaluator,
            "evaluate_robustness_report",
            None,
        )
        get_cached_evaluation_report = getattr(
            self.individual_evaluator,
            "get_cached_evaluation_report",
            None,
        )

        report = None
        if callable(get_cached_robustness_report):
            report = get_cached_robustness_report(individual, config)

        if report is None and force_robustness and callable(evaluate_robustness_report):
            try:
                report = evaluate_robustness_report(individual, config)
            except Exception as exc:
                logger.debug("summary 用 robustness 評価に失敗しました: %s", exc)

        if report is None and callable(get_cached_evaluation_report):
            report = get_cached_evaluation_report(individual)

        if (
            report is not None
            and is_evaluation_report(report)
            and report.metadata.get("evaluation_fidelity") == "coarse"
        ):
            report = None

        if report is None and is_multi_fidelity_enabled(config):
            try:
                self._evaluate_individual_with_full_fidelity(individual, config)
            except Exception as exc:
                logger.debug("summary 用 full 評価に失敗しました: %s", exc)
            if callable(get_cached_evaluation_report):
                report = get_cached_evaluation_report(individual)

        if not is_evaluation_report(report):
            return None

        if primary_fitness is None:
            fitness_score = extract_primary_fitness(individual)
            numeric_fitness = fitness_score if isfinite(fitness_score) else None
        else:
            numeric_fitness = (
                float(primary_fitness) if isfinite(float(primary_fitness)) else None
            )

        selection_rank = selection_rank_override
        if selection_rank is None:
            selection_rank = get_two_stage_rank(individual)

        selection_score: Any = selection_score_override
        if selection_score is None:
            selection_score = get_two_stage_score(individual)
        if not isinstance(selection_score, (tuple, list)):
            selection_score = None

        return build_report_summary(
            cast("EvaluationReport", report),  # type: ignore[reportArgumentType]
            selection_rank=selection_rank if isinstance(selection_rank, int) else None,
            selection_score=selection_score,
            fitness_score=numeric_fitness,
        )

    def _evaluate_individual_with_full_fidelity(
        self,
        individual: Any,
        config: GAConfig,
    ) -> Tuple[float, ...]:
        """必要に応じて full fidelity で個体を再評価する。"""
        if is_multi_fidelity_enabled(config):
            return self.individual_evaluator.evaluate(
                individual,
                config,
                force_refresh=True,
            )
        return self.individual_evaluator.evaluate(individual, config)

    @staticmethod
    def _extract_primary_fitness_from_result(result: Any) -> float:
        """評価結果から主 fitness を取り出す。"""
        return extract_primary_fitness_from_result(result)

    @staticmethod
    def _get_strategy_result_key(strategy: Any) -> str:
        """result 内部で戦略 summary を対応付けるキーを返す。"""
        strategy_id = getattr(strategy, "id", None)
        if strategy_id not in (None, ""):
            return str(strategy_id)
        return str(id(strategy))
