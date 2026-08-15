"""
遺伝的アルゴリズムエンジン

DEAPライブラリを使用したGA実装。
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from deap import tools

from app.services.auto_strategy.config.ga_config import GAConfig
from app.services.auto_strategy.genes.genetic_utils import GeneticUtils
from app.services.backtest.services.backtest_service import BacktestService

from ..evaluation.evaluation_fidelity import (
    build_coarse_ga_config,
    is_multi_fidelity_enabled,
)
from ..evaluation.individual_evaluator import IndividualEvaluator
from ..evaluation.parallel_evaluator import ParallelEvaluator
from ..fitness.fitness_sharing import FitnessSharing
from .deap_setup import DEAPSetup
from .evolution_runner import EvolutionRunner, EvolutionStoppedError
from .fitness_utils import (
    extract_result_fitness,
    sanitize_fitness_for_output,
)
from .ga_utils import (
    create_deap_mutate_wrapper,
    deap_crossover_strategy_genes,
)
from .parameter_tuning_manager import ParameterTuningManager
from .report_selection import (
    extract_primary_fitness,
)
from .result_processor import ResultProcessor

logger = logging.getLogger(__name__)

# 定数
DEFAULT_TIMEOUT_PER_INDIVIDUAL = 300.0
MAX_PERSISTED_POPULATION_SIZE = 100


class GeneticAlgorithmEngine:
    """遺伝的アルゴリズムエンジン。

    DEAPライブラリを使用して戦略の自動生成・最適化を行います。
    複雑な分離構造を削除し、直接的で理解しやすい実装に変更しました。
    """

    def __init__(
        self,
        backtest_service: BacktestService,
        gene_generator: Any,
        seed_strategy_provider: Any | None = None,
        builtin_seed_provider: Any | None = None,
    ):
        """初期化します。

        Args:
            backtest_service (BacktestService): バックテストサービス。
            gene_generator: 遺伝子生成器。
            seed_strategy_provider (Optional[Any]): 反復改善ループ用の
                シード戦略プロバイダ（get_seed_strategies(config) を実装）。
            builtin_seed_provider (Optional[Any]): 組み込みシード戦略プロバイダ
                （get_seed_strategies() を実装）。省略時は SeedStrategyFactory を
                使用する従来動作。
        """
        self.backtest_service = backtest_service
        self.gene_generator = gene_generator
        self.seed_strategy_provider = seed_strategy_provider
        self._builtin_seed_provider = builtin_seed_provider

        # 実行状態
        self._stop_event = threading.Event()

        # 分離されたコンポーネント
        self.deap_setup = DEAPSetup()
        self.result_processor = ResultProcessor()

        # 個体評価器（標準GAモード）
        logger.info("[Standard] 標準GAモードで起動")
        self.individual_evaluator = IndividualEvaluator(backtest_service)

        self.individual_class: Any = None  # setup_deap時に設定
        self.fitness_sharing: Any = None  # setup_deap時に初期化
        self.parameter_tuning_manager: ParameterTuningManager = ParameterTuningManager(
            self.individual_evaluator
        )

    def setup_deap(self, config: GAConfig) -> None:
        """
        DEAP フレームワークのコア設定（個体定義、演算子登録）を実行します。

        このメソッドは、`creator.create` を使用して以下の要素を動的に定義・登録します：
        1. 適応度クラス（`Fitness`）: 目的関数ごとの重みを設定。
        2. 個体クラス（`Individual`）: `StrategyGene` を継承し、適応度属性を持つクラス。
        3. 演算子（ツールボックス）:
           - 選択: NSGA-II。
           - 交叉: 遺伝子構造に対応した交叉アルゴリズム。
           - 突然変異: 適応的な突然変異率制御を含むラップされた演算子。
           - 評価: `IndividualEvaluator.evaluate` メソッドを登録。

        Args:
            config (GAConfig): 世代数、個体数、報酬設計、突然変異率、選択アルゴリズム等のGA設定。

        Note:
            このメソッドは `run_evolution` の内部で呼び出されますが、
            テストや特殊な実行環境で個別に設定を初期化する場合にも使用可能です。
        """
        # DEAP環境をセットアップ（戦略個体生成メソッドで統合）
        self.deap_setup.setup_deap(
            config,
            self._create_strategy_individual,
            self.individual_evaluator.evaluate,  # type: ignore[arg-type]  # 評価関数は(individual, config, ...)シグネチャ
            deap_crossover_strategy_genes,
        )

        # 個体クラスを取得（個体生成時に使用）
        self.individual_class = self.deap_setup.get_individual_class()

        # フィットネス共有の初期化
        fitness_sharing_config = config.fitness_sharing
        if fitness_sharing_config.enable_fitness_sharing:
            self.fitness_sharing = FitnessSharing(
                sharing_radius=fitness_sharing_config.sharing_radius,
                alpha=fitness_sharing_config.sharing_alpha,
                sampling_threshold=fitness_sharing_config.sampling_threshold,
                sampling_ratio=getattr(
                    fitness_sharing_config,
                    "sampling_ratio",
                    FitnessSharing.SAMPLING_RATIO,
                ),
            )
        else:
            self.fitness_sharing = None

        logger.info("DEAP環境のセットアップ完了")

    def run_evolution(
        self,
        config: GAConfig,
        backtest_config: dict[str, Any],
        progress_callback: Callable[[int, int, float | None], None] | None = None,
    ) -> dict[str, Any]:
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
            start_time = time.time()

            logger.info(
                "GA Engine - Starting evolution with backtest_config: %s",
                backtest_config,
            )

            self._raise_if_stop_requested("開始前")

            # グローバル乱数源をシードし、進化全体を再現可能にする
            # （初期集団生成・交叉・突然変異が `random` / `numpy.random` に依存）。
            self._seed_global_random(config)

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
            population = cast(Any, toolbox).population(n=config.population_size)

            # シード戦略の注入（ハイブリッド初期化）
            # 反復改善シード（過去の合格戦略）は use_seed_strategies に関係なく
            # 常に優先注入し、組み込みシードは注入率に従って注入する。
            previous_seeds = self._get_previous_seed_strategies(config)
            builtin_seeds = (
                self._get_builtin_seed_strategies(config)
                if config.use_seed_strategies
                else []
            )
            seeds = list(previous_seeds) + list(builtin_seeds)
            if seeds:
                individual_class = self.deap_setup.get_individual_class()
                if individual_class is not None:
                    num_builtin_to_inject = min(
                        int(len(population) * config.seed_injection_rate),
                        len(builtin_seeds),
                    )
                    num_to_inject = min(
                        len(previous_seeds) + num_builtin_to_inject,
                        len(seeds),
                        len(population),
                    )
                    for i in range(num_to_inject):
                        seed = seeds[i]
                        population[i] = individual_class(
                            **GeneticUtils.extract_gene_params(seed)
                        )
                    logger.info(
                        "シード戦略を %d 個注入しました "
                        "(反復改善 %d 件 + 組み込み %d 件, 注入率 %.1f%%)",
                        num_to_inject,
                        min(len(previous_seeds), num_to_inject),
                        max(0, num_to_inject - len(previous_seeds)),
                        config.seed_injection_rate * 100,
                    )
                else:
                    logger.warning(
                        "個体クラスが未初期化のため、シード戦略の注入をスキップしました"
                    )

            self._raise_if_stop_requested("初期集団生成後")

            # 適応的突然変異用mutate_wrapperの設定
            individual_class = self.deap_setup.get_individual_class()
            if individual_class is None:
                raise RuntimeError(
                    "Individual class must be initialized "
                    "before registering mutate operator."
                )
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
                    runner,
                    population,
                    config,
                    progress_callback=progress_callback,
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

    def _set_generator_context(self, backtest_config: dict[str, Any]) -> None:
        """ジェネレーターにコンテキストを設定します。

        Args:
            backtest_config (Dict[str, Any]): バックテスト設定。
        """
        try:
            tf = backtest_config.get("timeframe")
            sym = backtest_config.get("symbol")
            if hasattr(self.gene_generator, "smart_condition_generator"):
                smart_gen = self.gene_generator.smart_condition_generator
                if smart_gen and hasattr(smart_gen, "set_context"):
                    smart_gen.set_context(timeframe=tf, symbol=sym)
        except (AttributeError, TypeError) as e:
            logger.debug(f"コンテキスト設定スキップ: {e}")

    def _seed_global_random(self, config: GAConfig) -> None:
        """config.random_state に基づいてグローバル乱数源をシードする。

        遺伝子生成器・条件生成器・進化ループは `random` / `numpy.random` の
        グローバル状態に依存しているため、実行開始時に明示的にシードして
        再現可能な進化を保証する。None の場合は何もしない（従来動作）。
        """
        seed = getattr(config, "random_state", None)
        if seed is None:
            return
        seed_int = int(seed)
        random.seed(seed_int)
        np.random.seed(seed_int)
        logger.info("GA乱数シードを設定しました: %d", seed_int)

    def _get_builtin_seed_strategies(self, config: GAConfig) -> list[Any]:
        """組み込みシード戦略をシャッフルして返す。"""
        if self._builtin_seed_provider is not None:
            seeds = list(self._builtin_seed_provider.get_seed_strategies())
        else:
            # レイヤリング逆転を避けるため、generators への依存は factory 経由の
            # 注入が推奨。省略時は従来動作（遅延 import）を維持する。
            from app.services.auto_strategy.generators.seed_strategy_factory import (  # noqa: E501
                SeedStrategyFactory,
            )

            seeds = list(SeedStrategyFactory.get_all_seeds())

        if len(seeds) <= 1:
            return seeds

        seed = getattr(config, "random_state", None)
        if seed is not None:
            # NumPy scalar 由来の seed も Python の int に正規化して受ける。
            rng = random.Random(int(seed))
            rng.shuffle(seeds)
        else:
            random.shuffle(seeds)

        return seeds

    def _get_previous_seed_strategies(self, config: GAConfig) -> list[Any]:
        """反復改善ループ用の過去の合格戦略を取得する。

        無効な場合や取得失敗時は空リストを返します。
        過去戦略同士は注入順の偏りが出ないようシャッフルします。
        """
        iterative_config = getattr(config, "iterative_improvement_config", None)
        if (
            iterative_config is None
            or not getattr(iterative_config, "enabled", False)
            or self.seed_strategy_provider is None
        ):
            return []

        try:
            previous = list(self.seed_strategy_provider.get_seed_strategies(config))
        except Exception as exc:
            logger.warning(
                "反復改善シードの取得に失敗したため従来のシードのみ使用します: %s",
                exc,
            )
            return []

        if not previous:
            return []

        if len(previous) > 1:
            seed = getattr(config, "random_state", None)
            if seed is not None:
                # 組み込みシードと異なる順序になるよう seed+1 を使用
                rng = random.Random(int(seed) + 1)
                rng.shuffle(previous)
            else:
                random.shuffle(previous)

        logger.info(
            "反復改善: 過去の合格戦略 %d 件をシードとして優先注入します",
            len(previous),
        )
        return previous

    @staticmethod
    def _safe_fitness_stat(
        values: Any, reducer: Callable[[np.ndarray], float]
    ) -> float:
        """±inf（ペナルティ）を除いた集団 fitness の統計量を計算する。

        制約違反・低取引回数などの個体は fitness に ±inf（ペナルティ）が
        設定されるため、そのまま `np.mean` / `np.std` を計算すると
        RuntimeWarning（invalid value encountered in subtract）を引き起こす。
        非有限値はペナルティマーカーとして統計から除外する。
        """
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return float("nan")
        return float(reducer(finite))

    def _create_statistics(self) -> Any:
        """
        統計情報収集オブジェクトを作成

        世代ごとのフィットネスの平均、標準偏差、最小値、最大値を
        記録するためのDEAP Statisticsオブジェクトを構築します。

        Returns:
            統計情報収集用オブジェクト
        """
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", lambda v: self._safe_fitness_stat(v, np.mean))
        stats.register("std", lambda v: self._safe_fitness_stat(v, np.std))
        stats.register("min", lambda v: self._safe_fitness_stat(v, np.min))
        stats.register("max", lambda v: self._safe_fitness_stat(v, np.max))
        return stats

    def _create_parallel_evaluator(self, config: GAConfig) -> ParallelEvaluator | None:
        """並列評価器を作成します。

        Args:
            config (GAConfig): GA設定。

        Returns:
            ParallelEvaluator: 並列評価器インスタンス、またはNone。
        """
        evaluation_config = getattr(config, "evaluation_config", None)
        if evaluation_config is None or not getattr(
            evaluation_config, "enable_parallel", False
        ):
            return None

        from ..evaluation.evaluation_worker import (
            initialize_worker_process,
            worker_evaluate_individual,
        )

        # 並列ワーカー用のデータ準備
        worker_initargs: Any = ()

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
            evaluate_func=worker_evaluate_individual,  # type: ignore[arg-type]  # トップレベル関数を指定
            max_workers=getattr(evaluation_config, "max_workers", None),
            timeout_per_individual=getattr(
                evaluation_config, "timeout", DEFAULT_TIMEOUT_PER_INDIVIDUAL
            ),
            worker_initializer=initialize_worker_process,  # トップレベル関数を指定
            worker_initargs=worker_initargs,
        )
        return parallel_evaluator

    def _create_evolution_runner(
        self,
        toolbox: Any,
        stats: Any,
        population: Any = None,
        config: GAConfig | None = None,
        parallel_evaluator: ParallelEvaluator | None = None,
    ) -> EvolutionRunner:
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

    def _run_optimization(
        self,
        runner: EvolutionRunner,
        population: Any,
        config: GAConfig,
        progress_callback: Callable[[int, int, float | None], None] | None = None,
    ) -> tuple[Any, Any, Any]:
        """独立したEvolutionRunnerを使用して最適化アルゴリズムを実行します。

        Args:
            runner (EvolutionRunner): EvolutionRunnerインスタンス。
            population: 初期個体群。
            config (GAConfig): GA設定。
            progress_callback: 進捗通知コールバック（オプション）。

        Returns:
            tuple: 最適化後の個体群、ログブック、殿堂入りオブジェクト。
        """
        # Hall of Fame は常に ParetoFront を使う。
        # 目的関数が 1 つでも、同じ評価パイプラインで扱う。
        halloffame = tools.ParetoFront()

        # 統一された進化メソッドを実行
        population, logbook = runner.run_evolution(
            population,
            config,
            halloffame,
            should_stop=self._stop_event.is_set,
            progress_callback=progress_callback,
        )

        return population, logbook, halloffame

    def _process_results(
        self,
        population: Any,
        config: GAConfig,
        logbook: Any,
        start_time: float,
        halloffame: Any = None,
    ) -> dict[str, Any]:
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

        if best_evaluation_summary is None:
            best_evaluation_summary = (
                self.parameter_tuning_manager.build_individual_evaluation_summary(
                    best_gene,
                    config,
                )
            )

        execution_time = time.time() - start_time

        ranked_population = self.result_processor.sort_population(population)
        persisted_population = ranked_population[:MAX_PERSISTED_POPULATION_SIZE]

        # 最終的な結果の構築
        # fitness の ±inf / NaN（制約違反ペナルティ）は DEAP 内部専用の値であり、
        # DB 保存や API の JSON 応答で ValueError を引き起こすため、
        # 出力境界で安全な有限値（または None）へ置換する。
        best_fitness_value = sanitize_fitness_for_output(best_fitness_value)
        fitness_scores = [
            sanitize_fitness_for_output(extract_primary_fitness(individual)) or 0.0
            for individual in persisted_population
        ]

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

        result["all_strategies"] = persisted_population
        result["fitness_scores"] = fitness_scores
        result["evaluation_summaries"] = self._collect_population_evaluation_summaries(
            persisted_population,
            config,
        )
        # パレート解の fitness も出力境界で安全化する（リスト形状は維持）
        result["pareto_front"] = [
            {
                **solution,
                "fitness_values": [
                    (
                        float(v)
                        if isinstance(v, (int, float)) and np.isfinite(float(v))
                        else 0.0
                    )
                    for v in solution.get("fitness_values", [])
                ],
            }
            for solution in (best_strategies or [])
        ]
        result["objectives"] = config.objectives

        return result

    def stop_evolution(self) -> None:
        """進化を停止します。"""
        self._stop_event.set()

    def is_stop_requested(self) -> bool:
        """停止要求が出ているかを返します。"""
        return self._stop_event.is_set()

    def _raise_if_stop_requested(self, context: str = "") -> None:
        """停止要求がある場合は EvolutionStoppedError を送出します。"""
        if self._stop_event.is_set():
            if context:
                raise EvolutionStoppedError(f"停止要求により中断されました: {context}")
            raise EvolutionStoppedError("停止要求により中断されました")

    def _create_strategy_individual(self) -> Any:
        """戦略個体生成を行います。

        Returns:
            Individual: Individualオブジェクト。
        """
        try:
            # RandomGeneGeneratorを使用して遺伝子を生成
            gene = self.gene_generator.generate_random_gene()

            if self.individual_class is None:
                raise TypeError("個体クラス 'Individual' が初期化されていません。")

            # StrategyGeneのフィールドを使ってIndividualインスタンスを作成
            # IndividualはStrategyGeneを継承しているため、キーワード引数で初期化可能
            # asdictは再帰的に辞書化してしまうため使用しない
            return self.individual_class(**GeneticUtils.extract_gene_params(gene))

        except Exception as e:
            logger.error(f"個体生成中に致命的なエラーが発生しました: {e}")
            # 遺伝子生成はGAの根幹部分であり、失敗した場合は例外をスローして処理を停止するのが安全
            raise

    def _extract_result_best_fitness(
        self, best_individual: Any, config: GAConfig
    ) -> Any:
        """結果出力用の best fitness を抽出する。"""
        return extract_result_fitness(best_individual)

    def _collect_population_evaluation_summaries(
        self,
        population: list[Any],
        config: GAConfig,
    ) -> dict[str, dict[str, Any]]:
        """保存対象個体の評価 summary を収集する。"""
        summaries: dict[str, dict[str, Any]] = {}
        for individual in population:
            summary = self.parameter_tuning_manager.build_individual_evaluation_summary(
                individual, config
            )
            if not summary:
                continue
            strategy_key = self.result_processor.get_strategy_result_key(individual)
            summaries[strategy_key] = summary
        return summaries
