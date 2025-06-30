"""
遺伝的アルゴリズムエンジン

DEAPライブラリを使用したGA実装。
既存のBacktestServiceと統合し、戦略の自動生成・最適化を行います。
"""

import time
import logging
from typing import Dict, Any
from deap import tools


from ..models.ga_config import GAConfig
from ..factories.strategy_factory import StrategyFactory
from ..generators.random_gene_generator import RandomGeneGenerator
from app.core.services.backtest_service import BacktestService


# 分離されたモジュール
from .fitness_calculator import FitnessCalculator
from .deap_configurator import DEAPConfigurator
from .evolution_operators import EvolutionOperators
from .timeframe_manager import TimeframeManager
from .progress_manager import ProgressManager

logger = logging.getLogger(__name__)


class GeneticAlgorithmEngine:
    """
    遺伝的アルゴリズムエンジン

    DEAPライブラリを使用して戦略の自動生成・最適化を行います。
    各機能は専用モジュールに委譲し、メインの進化ループに集中します。
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
        """
        self.backtest_service = backtest_service
        self.strategy_factory = strategy_factory

        # 実行状態
        self.is_running = False

        # 分離されたコンポーネント
        self.gene_generator = gene_generator
        self.fitness_calculator = FitnessCalculator(backtest_service, strategy_factory)
        self.deap_configurator = DEAPConfigurator(self.gene_generator)
        self.evolution_operators = EvolutionOperators()
        self.timeframe_manager = TimeframeManager()
        self.progress_manager = ProgressManager()

        # DEAP関連
        self.toolbox = None
        self._fixed_backtest_config = None

    def setup_deap(self, config: GAConfig):
        """
        DEAP環境のセットアップ

        Args:
            config: GA設定
        """
        # DEAPConfigurator に委譲
        self.toolbox = self.deap_configurator.setup_deap_environment(
            config, self._evaluate_individual_wrapper
        )

        # 制約条件の適用
        self.deap_configurator.decorate_operators_with_constraints()

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

            # 進捗管理器の初期化
            self.progress_manager.set_start_time(start_time)
            self.progress_manager.set_current_generation(0)

            # バックテスト設定を保存
            logger.debug(f"GA実行開始時のバックテスト設定: {backtest_config}")

            # 評価環境固定化: GA実行開始時に一度だけバックテスト設定を決定
            if backtest_config:
                logger.info("評価環境を固定化中...")
                self._fixed_backtest_config = (
                    self.timeframe_manager.select_random_timeframe_config(
                        backtest_config
                    )
                )
                logger.debug(f"固定化された評価環境: {self._fixed_backtest_config}")
            else:
                self._fixed_backtest_config = None
                logger.info(
                    "バックテスト設定が提供されていないため、フォールバック設定を使用"
                )

            # DEAP環境のセットアップ
            self.setup_deap(config)

            # 初期個体群の生成と評価
            population = self._initialize_population(config)

            # 統計情報の取得
            stats = self.deap_configurator.get_statistics()
            logbook = self.deap_configurator.get_logbook()

            if stats is None or logbook is None:
                raise RuntimeError("統計情報が初期化されていません。")

            # 初期統計の記録
            record = stats.compile(population)
            logbook.record(gen=0, evals=len(population), **record)

            # 初期進捗通知
            self.progress_manager.notify_progress(
                config, population, backtest_config.get("experiment_id", "")
            )

            # 世代ループ: 指定された世代数だけ進化プロセスを繰り返す
            for generation in range(1, config.generations + 1):
                self.progress_manager.set_current_generation(generation)
                logger.info(f"世代 {generation}/{config.generations} 開始")

                # 進化演算の実行: 選択、交叉、突然変異、エリート保存など
                population = self._run_generation(population, config, self.toolbox)

                # 統計情報の記録: 各世代の個体群の適応度統計（平均、最大など）を記録
                record = stats.compile(population)
                logbook.record(gen=generation, evals=len(population), **record)

                # 進捗通知: 現在の世代の進捗状況を外部に通知
                self.progress_manager.notify_progress(
                    config, population, backtest_config.get("experiment_id", "")
                )

                logger.info(f"世代 {generation} 完了 - 最高適応度: {record['max']:.4f}")

            # 結果の整理: 最終世代の個体群から最も適応度の高い個体を選出
            best_individual = tools.selBest(population, 1)[0]

            # 遺伝子エンコーダーを使用してデコード: 数値リスト形式の遺伝子をStrategyGeneオブジェクトに変換
            from ..models.gene_encoding import GeneEncoder
            from ..models.strategy_gene import StrategyGene

            gene_encoder = GeneEncoder()
            best_gene = gene_encoder.decode_list_to_strategy_gene(
                best_individual, StrategyGene
            )

            execution_time = self.progress_manager.get_execution_time()

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

    def _initialize_population(self, config: GAConfig):
        """初期個体群の生成と評価"""
        if self.toolbox is None:
            raise RuntimeError("DEAP環境がセットアップされていません。")

        population = self.toolbox.population(n=config.population_size)  # type: ignore

        # 初期評価
        logger.info("初期個体群の評価開始...")
        fitnesses = self.toolbox.map(self.toolbox.evaluate, population)  # type: ignore
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        return population

    def _run_generation(self, population, config: GAConfig, toolbox):
        """単一世代の進化演算を実行"""
        # 親選択
        offspring = self.evolution_operators.select_parents(population, toolbox)

        # 交叉
        offspring = self.evolution_operators.perform_crossover(
            offspring, config.crossover_rate, toolbox
        )

        # 突然変異
        offspring = self.evolution_operators.perform_mutation(
            offspring, config.mutation_rate, toolbox
        )

        # 無効個体の評価
        offspring = self.evolution_operators.evaluate_invalid_individuals(
            offspring, toolbox
        )

        # エリート保存
        population = self.evolution_operators.apply_elitism(
            population, offspring, config.elite_size
        )

        return population

    def _evaluate_individual_wrapper(self, individual, config: GAConfig):
        """
        評価関数のラッパー

        FitnessCalculatorに委譲します。
        """
        return self.fitness_calculator.evaluate_individual(
            individual, config, self._fixed_backtest_config
        )

    def set_progress_callback(self, callback):
        """進捗コールバックを設定"""
        self.progress_manager.set_progress_callback(callback)

    def stop_evolution(self):
        """進化を停止"""
        self.is_running = False
        logger.info("進化停止が要求されました")
