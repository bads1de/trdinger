"""
遺伝的アルゴリズムエンジン

DEAPライブラリを使用したGA実装。
"""

import logging
import random
import time
from dataclasses import fields
from typing import Any, Dict, List, Optional

import numpy as np
from deap import base, creator, tools

from app.services.backtest.backtest_service import BacktestService

from ..config.ga import GAConfig
from ..generators.random_gene_generator import RandomGeneGenerator
from .fitness_sharing import FitnessSharing
from .genetic_operators import (
    create_deap_mutate_wrapper,
    crossover_strategy_genes,
    mutate_strategy_gene,
)
from .individual_evaluator import IndividualEvaluator
from .parallel_evaluator import ParallelEvaluator

logger = logging.getLogger(__name__)


class EvolutionRunner:
    """
    進化計算の実行を担当するクラス

    単一目的と多目的最適化のロジックをカプセル化した独立クラス。
    GAエンジンから分割されたが、現在は同一モジュール内で定義。
    並列評価をサポート。
    """

    def __init__(
        self,
        toolbox,
        stats,
        fitness_sharing: Optional[FitnessSharing] = None,
        population: Optional[List[Any]] = None,
        parallel_evaluator: Optional[ParallelEvaluator] = None,
    ):
        """
        初期化

        Args:
            toolbox: DEAPツールボックス
            stats: 統計情報収集オブジェクト
            fitness_sharing: 適応度共有オブジェクト（オプション）
            population: 個体集団（適応的突然変異用）
            parallel_evaluator: 並列評価器（オプション）
        """
        self.toolbox = toolbox
        self.stats = stats
        self.fitness_sharing = fitness_sharing
        self.population = population  # 適応的突然変異用
        self.parallel_evaluator = parallel_evaluator

    def run_evolution(
        self, population: List[Any], config: Any, halloffame: Optional[Any] = None
    ) -> tuple[List[Any], Any]:
        """
        進化アルゴリズムの実行（単一・多目的 統一版）

        単一目的・多目的を問わず、toolboxに登録された演算子と
        渡されたhalloffameオブジェクト（HallOfFame または ParetoFront）を使用して
        進化計算を実行します。

        Args:
            population: 初期個体群
            config: GA設定
            halloffame: 殿堂入り個体リスト（HallOfFame または ParetoFront）

        Returns:
            (最終個体群, 進化ログ)
        """
        logger.info(
            f"進化アルゴリズムを開始（世代数: {config.generations}, 目的数: {len(config.objectives)}）"
        )

        # 初期適応度評価
        population = self._evaluate_population(population)
        self._update_dynamic_objective_scalars(population, config)

        # Hall of Fame / Pareto Front 初回更新
        if halloffame is not None:
            halloffame.update(population)

        logbook = tools.Logbook()

        # 世代ループ
        for gen in range(config.generations):
            logger.debug(f"世代 {gen + 1}/{config.generations} を開始")

            # 適応度共有の適用（有効な場合、世代毎）
            if (
                getattr(config, "enable_fitness_sharing", False)
                and self.fitness_sharing
            ):
                population = self.fitness_sharing.apply_fitness_sharing(population)

            # 選択（親個体の選択）
            # cloneを使用することで、交叉・変異が元の個体に影響しないようにする
            offspring = list(self.toolbox.map(self.toolbox.clone, population))

            # 交叉
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

            # 未評価個体の評価（並列評価対応）
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            self._evaluate_invalid_individuals(invalid_ind)

            # 次世代の選択 (mu+lambda)
            # toolbox.select は DEAPSetup で NSGA-II などが登録されている
            population[:] = self.toolbox.select(offspring + population, len(population))

            self._update_dynamic_objective_scalars(population, config)

            # 統計の記録
            record = self.stats.compile(population) if self.stats else {}
            logbook.record(gen=gen, **record)

            # Hall of Fame / Pareto Front の更新
            if halloffame is not None:
                halloffame.update(population)

        logger.info("進化アルゴリズム完了")
        return population, logbook

    def _evaluate_population(self, population: List[Any]) -> List[Any]:
        """
        個体群の適応度評価（並列評価対応）

        Args:
            population: 評価対象の個体群

        Returns:
            評価された個体群
        """
        if self.parallel_evaluator:
            # 並列評価
            fitnesses = self.parallel_evaluator.evaluate_population(population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
        else:
            # シーケンシャル評価（フォールバック）
            fitnesses = list(self.toolbox.map(self.toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

        return population

    def _evaluate_invalid_individuals(self, invalid_ind: List[Any]) -> None:
        """
        適応度が無効な個体のみを評価（並列評価対応）

        Args:
            invalid_ind: 評価対象の無効な個体リスト
        """
        if not invalid_ind:
            return

        if self.parallel_evaluator:
            # 並列評価
            fitnesses = self.parallel_evaluator.evaluate_population(invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
        else:
            # シーケンシャル評価（フォールバック）
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

    def _update_dynamic_objective_scalars(
        self, population: List[Any], config: Any
    ) -> None:
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


class DEAPSetup:
    """
    DEAP設定クラス

    DEAPライブラリの設定とツールボックスの初期化を担当します。
    """

    def __init__(self):
        """初期化"""
        self.toolbox: Optional[base.Toolbox] = None
        self.Individual = None

    def setup_deap(
        self,
        config: GAConfig,
        create_individual_func,
        evaluate_func,
        crossover_func,
        mutate_func,
    ):
        """
        DEAP環境のセットアップ（多目的最適化専用）

        Args:
            config: GA設定
            create_individual_func: 個体生成関数
            evaluate_func: 評価関数
            crossover_func: 交叉関数
            mutate_func: 突然変異関数
        """
        # 多目的最適化用フィットネスクラスの定義
        fitness_class_name = "FitnessMulti"
        weights = tuple(config.objective_weights)
        logger.info(f"多目的最適化モード: 目的={config.objectives}, 重み={weights}")

        # 既存のフィットネスクラスを削除（再定義のため）
        if hasattr(creator, fitness_class_name):
            delattr(creator, fitness_class_name)

        # フィットネスクラスを作成
        creator.create(fitness_class_name, base.Fitness, weights=weights)
        fitness_class = getattr(creator, fitness_class_name)

        # 個体クラスの定義
        if hasattr(creator, "Individual"):
            delattr(creator, "Individual")

        from ..genes import StrategyGene

        # StrategyGeneを継承し、fitness属性を持つクラスを作成
        creator.create("Individual", StrategyGene, fitness=fitness_class)  # type: ignore
        self.Individual = creator.Individual  # type: ignore # 生成したクラスをインスタンス変数に格納

        # ツールボックスの初期化
        self.toolbox = base.Toolbox()

        # 個体生成関数の登録
        self.toolbox.register("individual", create_individual_func)
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual,  # type: ignore
        )

        # 評価関数の登録
        self.toolbox.register("evaluate", evaluate_func, config=config)

        # 進化演算子の登録（戦略遺伝子レベル）
        self.toolbox.register("mate", crossover_func, config=config)

        # 突然変異の登録（DEAP互換の返り値 (ind,) を保証するラッパー）
        def _mutate_wrapper(individual):
            res = mutate_func(individual, mutation_rate=config.mutation_rate)
            if isinstance(res, tuple):
                return res
            return (res,)

        self.toolbox.register("mutate", _mutate_wrapper)

        # 選択アルゴリズムの登録（目的数に応じて切り替え）
        if config.enable_multi_objective:
            self.toolbox.register("select", tools.selNSGA2)
            logger.info("多目的最適化モード: NSGA-II選択アルゴリズムを登録")
        else:
            # 単一目的の場合はトーナメント選択（デフォルトサイズ3）
            tourn_size = getattr(config, "tournament_size", 3)
            self.toolbox.register("select", tools.selTournament, tournsize=tourn_size)
            logger.info(
                f"単一目的最適化モード: トーナメント選択アルゴリズム(size={tourn_size})を登録"
            )

        logger.info("DEAP環境のセットアップ完了")

    def get_toolbox(self) -> Optional[base.Toolbox]:
        """ツールボックスを取得"""
        return self.toolbox

    def get_individual_class(self):
        """個体クラスを取得"""
        return self.Individual


class EvaluatorWrapper:
    """評価関数のラッパー（Pickle化対応）"""

    def __init__(self, evaluator, config):
        self.evaluator = evaluator
        self.config = config

    def __call__(self, individual):
        return self.evaluator.evaluate_individual(individual, self.config)


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
            logger.info("🔬 ハイブリッドGA+MLモードで起動")
            from .hybrid_individual_evaluator import HybridIndividualEvaluator

            self.individual_evaluator = HybridIndividualEvaluator(
                backtest_service=backtest_service,
                predictor=hybrid_predictor,
                feature_adapter=hybrid_feature_adapter,
            )
        else:
            logger.info("🧬 標準GAモードで起動")
            self.individual_evaluator = IndividualEvaluator(backtest_service)

        self.individual_class = None  # setup_deap時に設定
        self.fitness_sharing = None  # setup_deap時に初期化

    def setup_deap(self, config: GAConfig):
        """DEAP環境のセットアップ（統合版）を行います。

        Args:
            config (GAConfig): GA設定。
        """
        # DEAP環境をセットアップ（戦略個体生成メソッドで統合）
        self.deap_setup.setup_deap(
            config,
            self._create_strategy_individual,
            self.individual_evaluator.evaluate_individual,
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
        """進化アルゴリズムを実行します。

        独立したEvolutionRunnerを使って設定に応じて適切な最適化アルゴリズムを呼び出します。

        Args:
            config (GAConfig): GA設定。
            backtest_config (Dict[str, Any]): バックテスト設定。

        Returns:
            Dict[str, Any]: 進化結果。
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

            # 適応的突然変異用mutate_wrapperの設定
            individual_class = self.deap_setup.get_individual_class()
            mutate_wrapper = create_deap_mutate_wrapper(
                individual_class, population, config
            )
            toolbox.register("mutate", mutate_wrapper)

            # 独立したEvolutionRunnerの作成（並列評価対応）
            runner = self._create_evolution_runner(toolbox, stats, population, config)

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
        """統計情報収集オブジェクトを作成します。

        Returns:
            DEAP Statistics: 統計情報収集オブジェクト。
        """
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        return stats

    def _create_evolution_runner(self, toolbox, stats, population=None, config=None):
        """独立したEvolutionRunnerインスタンスを作成します。

        Args:
            toolbox: DEAPツールボックス。
            stats: 統計情報オブジェクト。
            population: 初期個体群（オプション）。
            config: GA設定（並列評価用）。

        Returns:
            EvolutionRunner: EvolutionRunnerインスタンス。
        """
        fitness_sharing = (
            self.fitness_sharing
            if hasattr(self, "fitness_sharing") and self.fitness_sharing
            else None
        )

        # 並列評価器の作成（設定で有効な場合）
        parallel_evaluator = None
        if config and getattr(config, "enable_parallel_evaluation", False):
            # 並列ワーカー用のデータ準備
            worker_initializer = None
            worker_initargs = ()

            try:
                # バックテスト設定が存在する場合のみデータを準備
                if (
                    hasattr(self.individual_evaluator, "_fixed_backtest_config")
                    and self.individual_evaluator._fixed_backtest_config
                ):
                    bc = self.individual_evaluator._fixed_backtest_config

                    # メインデータを取得（キャッシュになければロードされる）
                    main_data = self.individual_evaluator._get_cached_data(bc)

                    # 1分足データを取得（存在する場合）
                    minute_data = self.individual_evaluator._get_cached_minute_data(bc)

                    data_context = {"main_data": main_data}
                    if minute_data is not None:
                        data_context["minute_data"] = minute_data

                    from .parallel_evaluator import initialize_worker

                    worker_initializer = initialize_worker
                    worker_initargs = (data_context,)

                    logger.info("並列ワーカー用の共有データを準備しました")
            except Exception as e:
                logger.warning(
                    f"並列ワーカー用データ準備中にエラーが発生しました（データ共有なしで続行）: {e}"
                )

            parallel_evaluator = ParallelEvaluator(
                evaluate_func=EvaluatorWrapper(self.individual_evaluator, config),
                max_workers=getattr(config, "max_evaluation_workers", None),
                timeout_per_individual=getattr(config, "evaluation_timeout", 300.0),
                worker_initializer=worker_initializer,
                worker_initargs=worker_initargs,
            )
            logger.info(
                f"⚡ 並列評価有効: max_workers={parallel_evaluator.max_workers}"
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

    def _extract_best_individuals(self, population, config: GAConfig, halloffame=None):
        """最良個体を抽出し、デコードします。

        Args:
            population: 最終個体群。
            config (GAConfig): GA設定。
            halloffame: 殿堂入りオブジェクト（HallOfFame または ParetoFront）。

        Returns:
            tuple: 最良個体、最良遺伝子、および最良戦略のタプル。
        """
        from ..genes import StrategyGene
        from ..serializers.serialization import GeneSerializer

        gene_serializer = GeneSerializer()

        best_strategies = None
        best_individual = None

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
                    gene = gene_serializer.from_list(ind, StrategyGene)

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
            best_gene = gene_serializer.from_list(best_individual, StrategyGene)

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
            gene_dict = {f.name: getattr(gene, f.name) for f in fields(gene)}
            return self.individual_class(**gene_dict)

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

            logger.info("🔧 エリート個体のパラメータチューニングを開始")

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

            logger.info("✅ パラメータチューニング完了")
            return tuned_gene

        except Exception as e:
            logger.warning(f"パラメータチューニング中にエラーが発生: {e}")
            # エラー時は元の遺伝子を返す
            return best_gene
