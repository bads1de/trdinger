"""
遺伝的アルゴリズムエンジン

DEAPライブラリを使用したGA実装。
"""

import time
import logging
import numpy as np
from typing import Dict, Any, Optional
from deap import base, creator, tools, algorithms

from ..models.ga_config import GAConfig
from ..factories.strategy_factory import StrategyFactory
from ..generators.random_gene_generator import RandomGeneGenerator
from app.core.services.backtest_service import BacktestService

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

        # DEAP関連
        self.toolbox: Optional[base.Toolbox] = None
        self._fixed_backtest_config = None
        self.Individual = None  # 個体クラスを保持する変数を追加

    def setup_deap(self, config: GAConfig):
        """
        DEAP環境のセットアップ（統合版）

        Args:
            config: GA設定
        """
        # フィットネスクラスの定義（最大化問題）
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        # 個体クラスの定義
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)  # type: ignore
        self.Individual = creator.Individual  # type: ignore # 生成したクラスをインスタンス変数に格納

        # ツールボックスの初期化
        self.toolbox = base.Toolbox()

        # 個体生成関数の登録
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual  # type: ignore
        )

        # 評価関数の登録
        self.toolbox.register("evaluate", self._evaluate_individual, config=config)

        # 進化演算子の登録（戦略遺伝子レベル）
        self.toolbox.register("mate", self._crossover_strategy_genes)
        self.toolbox.register(
            "mutate", self._mutate_strategy_gene, mutation_rate=config.mutation_rate
        )
        self.toolbox.register("select", tools.selTournament, tournsize=3)

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
            self._fixed_backtest_config = self._select_timeframe_config(backtest_config)

            # DEAP環境のセットアップ
            self.setup_deap(config)

            # 初期個体群の生成
            # self.toolboxがNoneでないことをアサートし、静的解析エラーを回避
            assert self.toolbox is not None, "Toolbox must be initialized before use."
            # deap.Toolboxは動的に属性を登録するため、静的解析ではpopulation属性を検出できない
            population = self.toolbox.population(n=config.population_size)  # type: ignore

            # 統計情報の設定
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)

            # 進化アルゴリズムの実行
            population, logbook = algorithms.eaSimple(
                population,
                self.toolbox,
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
            from ..models.strategy_gene import StrategyGene

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

    def _create_individual(self):
        """個体生成（統合版）"""
        try:
            # RandomGeneGeneratorを使用して遺伝子を生成
            gene = self.gene_generator.generate_random_gene()

            # 遺伝子をエンコード
            from ..models.gene_encoding import GeneEncoder

            gene_encoder = GeneEncoder()
            encoded_gene = gene_encoder.encode_strategy_gene_to_list(gene)

            if not self.Individual:
                raise TypeError("個体クラス 'Individual' が初期化されていません。")
            return self.Individual(encoded_gene)

        except Exception as e:
            logger.error(f"個体生成中に致命的なエラーが発生しました: {e}")
            # 遺伝子生成はGAの根幹部分であり、失敗した場合は例外をスローして処理を停止するのが安全
            raise

    def _evaluate_individual(self, individual, config: GAConfig):
        """
        個体評価（統合版）

        Args:
            individual: 評価する個体
            config: GA設定

        Returns:
            フィットネス値のタプル
        """
        try:
            # 遺伝子デコード
            from ..models.gene_encoding import GeneEncoder
            from ..models.strategy_gene import StrategyGene

            gene_encoder = GeneEncoder()
            gene = gene_encoder.decode_list_to_strategy_gene(individual, StrategyGene)

            # バックテスト実行用の設定を構築
            backtest_config = (
                self._fixed_backtest_config.copy()
                if self._fixed_backtest_config
                else {}
            )

            # 戦略設定を追加（test_strategy_generationと同じ形式）
            backtest_config["strategy_config"] = {
                "strategy_type": "GENERATED_GA",
                "parameters": {"strategy_gene": gene.to_dict()},
            }

            # デバッグログ: 取引量設定を確認
            risk_management = gene.risk_management
            position_size = risk_management.get("position_size", 0.1)
            logger.debug(
                f"GA個体評価 - position_size: {position_size}, gene_id: {gene.id}"
            )

            result = self.backtest_service.run_backtest(backtest_config)

            # フィットネス計算
            fitness = self._calculate_fitness(result, config)

            return (fitness,)

        except Exception as e:
            logger.error(f"個体評価エラー: {e}")
            return (0.0,)

    def _calculate_fitness(
        self, backtest_result: Dict[str, Any], config: GAConfig
    ) -> float:
        """
        フィットネス計算（統合版）

        Args:
            backtest_result: バックテスト結果
            config: GA設定

        Returns:
            フィットネス値
        """
        try:
            # performance_metricsから基本メトリクスを取得
            performance_metrics = backtest_result.get("performance_metrics", {})

            total_return = performance_metrics.get("total_return", 0.0)
            sharpe_ratio = performance_metrics.get("sharpe_ratio", 0.0)
            max_drawdown = performance_metrics.get("max_drawdown", 1.0)
            win_rate = performance_metrics.get("win_rate", 0.0)
            total_trades = performance_metrics.get("total_trades", 0)

            # デバッグログ: メトリクス値を確認
            # logger.debug(
            #     f"フィットネス計算 - return: {total_return}, sharpe: {sharpe_ratio}, drawdown: {max_drawdown}, win_rate: {win_rate}, trades: {total_trades}"
            # )

            # 取引回数が0の場合は低いフィットネス値を返す
            if total_trades == 0:
                logger.warning("取引回数が0のため、低いフィットネス値を設定")
                return 0.1  # 完全に0ではなく、わずかな値を返す

            # 制約チェック
            if total_return < 0 or sharpe_ratio < config.fitness_constraints.get(
                "min_sharpe_ratio", 0
            ):
                return 0.0

            # 重み付きフィットネス計算
            fitness = (
                config.fitness_weights["total_return"] * total_return
                + config.fitness_weights["sharpe_ratio"] * sharpe_ratio
                + config.fitness_weights["max_drawdown"] * (1 - max_drawdown)
                + config.fitness_weights["win_rate"] * win_rate
            )

            return max(0.0, fitness)

        except Exception as e:
            logger.error(f"フィットネス計算エラー: {e}")
            return 0.0

    def _select_timeframe_config(
        self, backtest_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        タイムフレーム設定の選択（統合版）

        Args:
            backtest_config: バックテスト設定

        Returns:
            選択されたタイムフレーム設定
        """
        if not backtest_config:
            return {}

        # 簡単な実装: 設定をそのまま返す
        return backtest_config.copy()

    def stop_evolution(self):
        """進化を停止"""
        self.is_running = False

    def _crossover_strategy_genes(self, ind1, ind2):
        """
        戦略遺伝子レベルの交叉

        Args:
            ind1: 個体1（エンコードされた戦略遺伝子）
            ind2: 個体2（エンコードされた戦略遺伝子）

        Returns:
            交叉後の個体のタプル
        """
        try:
            # 遺伝子デコード
            from ..models.gene_encoding import GeneEncoder
            from ..models.strategy_gene import StrategyGene, crossover_strategy_genes

            gene_encoder = GeneEncoder()
            gene1 = gene_encoder.decode_list_to_strategy_gene(ind1, StrategyGene)
            gene2 = gene_encoder.decode_list_to_strategy_gene(ind2, StrategyGene)

            # 戦略遺伝子レベルの交叉
            child1, child2 = crossover_strategy_genes(gene1, gene2)

            # 再エンコード
            encoded_child1 = gene_encoder.encode_strategy_gene_to_list(child1)
            encoded_child2 = gene_encoder.encode_strategy_gene_to_list(child2)

            # 個体を更新
            ind1[:] = encoded_child1
            ind2[:] = encoded_child2

            return ind1, ind2

        except Exception as e:
            logger.error(f"戦略遺伝子交叉エラー: {e}")
            # エラー時は元の個体をそのまま返す
            return ind1, ind2

    def _mutate_strategy_gene(self, individual, mutation_rate: float = 0.1):
        """
        戦略遺伝子レベルの突然変異

        Args:
            individual: 個体（エンコードされた戦略遺伝子）
            mutation_rate: 突然変異率

        Returns:
            突然変異後の個体のタプル
        """
        try:
            # 遺伝子デコード
            from ..models.gene_encoding import GeneEncoder
            from ..models.strategy_gene import StrategyGene, mutate_strategy_gene

            gene_encoder = GeneEncoder()
            gene = gene_encoder.decode_list_to_strategy_gene(individual, StrategyGene)

            # 戦略遺伝子レベルの突然変異
            mutated_gene = mutate_strategy_gene(gene, mutation_rate)

            # 再エンコード
            encoded_mutated = gene_encoder.encode_strategy_gene_to_list(mutated_gene)

            # 個体を更新
            individual[:] = encoded_mutated

            return (individual,)

        except Exception as e:
            logger.error(f"戦略遺伝子突然変異エラー: {e}")
            # エラー時は元の個体をそのまま返す
            return (individual,)
