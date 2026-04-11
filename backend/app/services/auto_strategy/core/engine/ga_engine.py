"""
遺伝的アルゴリズムエンジン

DEAPライブラリを使用したGA実装。
"""

from __future__ import annotations

import logging
import threading
import time

from typing import Any, Dict, List, Optional, Tuple


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
from app.services.backtest.services.backtest_service import BacktestService

from ..evaluation.individual_evaluator import IndividualEvaluator
from ..evaluation.parallel_evaluator import ParallelEvaluator
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
    extract_result_fitness,
)
from .parameter_tuning_manager import ParameterTuningManager
from .report_selection import (
    extract_primary_fitness,
)
from .result_processor import ResultProcessor

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
        self.result_processor = ResultProcessor()

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
        self.parameter_tuning_manager: ParameterTuningManager = ParameterTuningManager(
            self.individual_evaluator
        )

    def setup_deap(self, config: GAConfig) -> None:
        """
        DEAP フレームワークのコア設定（個体定義、演算子登録）を実行します。

        このメソッドは、`creator.create` を使用して以下の要素を動的に定義・登録します：
        1. 適応度クラス（`Fitness`）: 多目的または単一目的の重みを設定。
        2. 個体クラス（`Individual`）: `StrategyGene` を継承し、適応度属性を持つクラス。
        3. 演算子（ツールボックス）: 
           - 選択: NSGA-II（多目的）またはトーナメント選択（単一目的）。
           - 交叉: 遺伝子構造に対応した交叉アルゴリズム。
           - 突然変異: 適応的な突然変異率制御を含むラップされた演算子。
           - 評価: `IndividualEvaluator.evaluate` メソッドを登録。

        Args:
            config (GAConfig): 世代数、個体数、報酬設計、突然変異率、選択アルゴリズム等のGA設定。

        Note:
            このメソッドは `run_evolution` の内部で呼び出されますが、
            テストや特殊な実行環境で個別に設定を初期化する場合にも使用可能です。
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
                sampling_threshold=fitness_sharing_config.get(
                    "sampling_threshold", 200
                ),
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
        進化計算プロセスを開始し、最適な取引戦略を探索します。

        このメソッドはエンジンのメインエントリーポイントであり、以下のプロセスを実行します：
        1. 実行状態の初期化と事前チェック（ストップリクエストの確認）。
        2. バックテストデータの準備と評価器（Evaluator）への設定。
        3. DEAPフレームワークのセットアップ（`setup_deap`）。
        4. 初期集団（Population）の生成。必要に応じてシード戦略を注入。
        5. 世代交代ループの実行（`EvolutionRunner` への委譲）:
           - 各個体の評価（並列実行をサポート）。
           - 選択・交叉・突然変異による次世代の生成。
           - 統計情報の記録とホール・オブ・フェイム（最良個体群）の更新。
        6. 最良個体群の抽出と最終レポートの生成。

        Args:
            config (GAConfig): アルゴリズムのハイパーパラメータ（世代数、集団サイズ、交叉率等）。
            backtest_config (Dict[str, Any]): 評価に使用する市場データの設定（銘柄、期間、証拠金等）。

        Returns:
            Dict[str, Any]: 探索結果を含む辞書。以下のキーを含みます：
                - "best_individual": 最も適応度の高かった個体の遺伝子。
                - "best_individuals": ホール・オブ・フェイムに記録された優良個体リスト。
                - "logbook": 各世代の統計情報（平均、最大、最小適応度等）。
                - "metadata": 実行時間、キャッシュヒット率、終了理由等の付随情報。

        Raises:
            EvolutionStoppedError: 外部からの停止リクエスト（`stop()` メソッドの呼び出し）により中断された場合。
            RuntimeError: DEAPのセットアップに失敗した場合や、致命的な実行エラーが発生した場合。

        Note:
            - スレッドセーフ: `stop()` メソッドを通じて他スレッドから安全に停止させることが可能です。
            - キャッシュ: `IndividualEvaluator` 内のキャッシュにより、同一世代や世代間での重複評価を最小限に抑えます。
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
        evaluation_config = getattr(config, "evaluation_config", None)
        if evaluation_config is None or not getattr(evaluation_config, "enable_parallel", False):
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
            max_workers=getattr(evaluation_config, "max_workers", None),
            timeout_per_individual=getattr(evaluation_config, "timeout", 300.0),
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
        best_individual, best_gene, best_strategies = (
            self.result_processor.extract_best_individuals(
                population, config, halloffame
            )
        )

        if best_individual is not None and is_multi_fidelity_enabled(config):
            try:
                refreshed = self.parameter_tuning_manager.evaluate_individual_with_full_fidelity(
                    best_individual,
                    config,
                )
                if getattr(best_individual, "fitness", None) is not None:
                    best_individual.fitness.values = tuple(refreshed)
            except Exception as exc:
                logger.warning("最終候補の full 評価に失敗しました: %s", exc)

        best_fitness_value = self._extract_result_best_fitness(best_individual, config)
        best_evaluation_summary = (
            self.parameter_tuning_manager.build_individual_evaluation_summary(
                best_individual, config
            )
        )

        # パラメータチューニング（有効な場合）
        if config.tuning_config.enabled:
            (
                best_gene,
                best_fitness_value,
                best_evaluation_summary,
            ) = self.parameter_tuning_manager.tune_and_select_best_gene(
                population=population,
                current_best_gene=best_gene,
                config=config,
                fallback_fitness=best_fitness_value,
                fallback_summary=best_evaluation_summary,
            )
        elif best_evaluation_summary is None:
            best_evaluation_summary = (
                self.parameter_tuning_manager.build_individual_evaluation_summary(
                    best_gene,
                    config,
                    force_robustness=bool(config.two_stage_selection_config.enabled),
                )
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
            ranked_population = self.result_processor.rank_population_for_persistence(
                population
            )
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

    def _extract_result_best_fitness(
        self, best_individual: Any, config: GAConfig
    ) -> object:
        """結果出力用の best fitness を抽出する。"""
        return extract_result_fitness(
            best_individual,
            enable_multi_objective=config.enable_multi_objective,
        )

    def _collect_population_evaluation_summaries(
        self,
        population: List[Any],
        config: GAConfig,
    ) -> Dict[str, Dict[str, Any]]:
        """保存対象個体の評価 summary を収集する。"""
        summaries: Dict[str, Dict[str, Any]] = {}
        for individual in population:
            summary = self.parameter_tuning_manager.build_individual_evaluation_summary(
                individual, config
            )
            if not summary:
                continue
            strategy_key = self.result_processor.get_strategy_result_key(individual)
            summaries[strategy_key] = summary
        return summaries

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
        return self.parameter_tuning_manager.build_individual_evaluation_summary(
            individual,
            config,
            force_robustness=force_robustness,
            primary_fitness=primary_fitness,
            selection_rank_override=selection_rank_override,
            selection_score_override=selection_score_override,
        )

    def _evaluate_individual_with_full_fidelity(
        self,
        individual: Any,
        config: GAConfig,
    ) -> Tuple[float, ...]:
        """必要に応じて full fidelity で個体を再評価する。"""
        return self.parameter_tuning_manager.evaluate_individual_with_full_fidelity(
            individual, config
        )

    def _extract_primary_fitness_from_result(self, result: Any) -> float:
        """評価結果から主 fitness を取り出す。"""
        return self.parameter_tuning_manager.extract_primary_fitness_from_result(result)

    def _get_strategy_result_key(self, strategy: Any) -> str:
        """result 内部で戦略 summary を対応付けるキーを返す。"""
        return self.result_processor.get_strategy_result_key(strategy)
