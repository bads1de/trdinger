"""
遺伝的アルゴリズムエンジン

DEAPライブラリを使用したGA実装。
既存のBacktestServiceと統合し、戦略の自動生成・最適化を行います。
"""

import random
import time
import multiprocessing
from typing import List, Dict, Any, Callable, Optional, Tuple
import logging
import numpy as np
from datetime import datetime, timedelta, timezone

from deap import base, creator, tools

from ..models.strategy_gene import (
    encode_gene_to_list,
    decode_list_to_gene,
)
from ..models.ga_config import GAConfig, GAProgress
from ..factories.strategy_factory import StrategyFactory
from ..generators.random_gene_generator import RandomGeneGenerator
from app.core.services.backtest_service import BacktestService

logger = logging.getLogger(__name__)


class GeneticAlgorithmEngine:
    """
    遺伝的アルゴリズムエンジン

    DEAPライブラリを使用して戦略の自動生成・最適化を行います。
    """

    def __init__(
        self, backtest_service: BacktestService, strategy_factory: StrategyFactory
    ):
        """
        初期化

        Args:
            backtest_service: バックテストサービス
            strategy_factory: 戦略ファクトリー
        """
        self.backtest_service = backtest_service
        self.strategy_factory = strategy_factory
        self.toolbox = None
        self.progress_callback: Optional[Callable] = None

        # 新しいランダム遺伝子生成器
        self.gene_generator = RandomGeneGenerator()

        # 統計情報
        self.stats = None
        self.logbook = None

        # 実行状態
        self.is_running = False
        self.current_generation = 0
        self.start_time = 0

        # 利用可能な時間足
        self.available_timeframes = ["15m", "30m", "1h", "4h", "1d"]

    def setup_deap(self, config: GAConfig):
        """
        DEAP環境のセットアップ

        Args:
            config: GA設定
        """
        # フィットネスクラスの定義（最大化問題）
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        # 個体クラスの定義
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        # ツールボックスの初期化
        self.toolbox = base.Toolbox()

        # 遺伝子長の計算（v1仕様: 5指標×2 + エントリー条件3 + イグジット条件3 = 16）
        config.max_indicators * 2 + 6

        # 個体生成関数（新しいランダム遺伝子生成器を使用）
        def create_individual():
            """新しいランダム遺伝子生成器を使用して個体を生成"""
            try:
                # ランダム遺伝子生成器で戦略遺伝子を生成
                gene = self.gene_generator.generate_random_gene()

                # 戦略遺伝子を数値リストにエンコード
                individual = encode_gene_to_list(gene)

                return creator.Individual(individual)
            except Exception as e:
                logger.warning(f"新しい遺伝子生成に失敗、フォールバックを使用: {e}")
                # フォールバック: 従来の方法
                individual = []

                # 指標部分（最低1個の指標を保証）
                for i in range(config.max_indicators):
                    if i == 0:
                        # 最初の指標は必ず有効にする
                        indicator_id = random.uniform(0.1, 0.9)  # 0を避ける
                    else:
                        # 他の指標は50%の確率で有効
                        indicator_id = random.uniform(0.0, 1.0)

                    param_val = random.uniform(0.0, 1.0)
                    individual.extend([indicator_id, param_val])

                # 条件部分
                for _ in range(6):  # エントリー3 + エグジット3
                    individual.append(random.uniform(0.0, 1.0))

                return creator.Individual(individual)

        self.toolbox.register("individual", create_individual)
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        # 評価関数（バックテスト設定は run_evolution で設定済み）
        # self._current_backtest_config = None  # リセットしない
        self.toolbox.register(
            "evaluate", self._evaluate_individual_wrapper, config=config
        )

        # 選択・交叉・突然変異
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        # 制約条件の適用
        self.toolbox.decorate("mate", self._apply_constraints)
        self.toolbox.decorate("mutate", self._apply_constraints)

        # 並列処理の設定
        if config.parallel_processes:
            pool = multiprocessing.Pool(config.parallel_processes)
            self.toolbox.register("map", pool.map)

        # 統計情報の設定
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

        # ログブックの初期化
        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "evals", "std", "min", "avg", "max"

        logger.info("DEAP環境のセットアップ完了")

    def _select_random_timeframe_config(
        self, base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        ランダムな時間足を選択し、適切な期間を設定

        Args:
            base_config: ベースとなるバックテスト設定

        Returns:
            時間足と期間が調整された設定
        """
        # ランダムに時間足を選択
        selected_timeframe = random.choice(self.available_timeframes)
        logger.info(f"ランダム時間足選択: {selected_timeframe}")

        # 現在時刻を基準に期間を設定
        end_date = datetime.now(timezone.utc)

        # 時間足に応じて適切な期間を設定
        if selected_timeframe == "15m":
            start_date = end_date - timedelta(days=7)  # 15分足: 1週間
        elif selected_timeframe == "30m":
            start_date = end_date - timedelta(days=14)  # 30分足: 2週間
        elif selected_timeframe == "1h":
            start_date = end_date - timedelta(days=30)  # 1時間足: 1ヶ月
        elif selected_timeframe == "4h":
            start_date = end_date - timedelta(days=60)  # 4時間足: 2ヶ月
        else:  # 1d
            start_date = end_date - timedelta(days=90)  # 日足: 3ヶ月

        # 設定をコピーして更新
        config = base_config.copy()
        config["timeframe"] = selected_timeframe
        config["start_date"] = start_date.isoformat()
        config["end_date"] = end_date.isoformat()

        return config

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
            self.start_time = time.time()
            self.current_generation = 0

            # バックテスト設定を保存
            logger.info(f"GA実行開始時のバックテスト設定: {backtest_config}")
            self._current_backtest_config = backtest_config
            logger.info(
                f"保存後の_current_backtest_config: {self._current_backtest_config}"
            )

            # DEAP環境のセットアップ
            self.setup_deap(config)

            # 初期個体群の生成
            population = self.toolbox.population(n=config.population_size)

            # 初期評価
            logger.info("初期個体群の評価開始...")
            fitnesses = self.toolbox.map(self.toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # 統計情報の記録
            record = self.stats.compile(population)
            self.logbook.record(gen=0, evals=len(population), **record)

            # 進捗コールバック
            if self.progress_callback:
                progress = self._create_progress_info(
                    config, population, backtest_config.get("experiment_id", "")
                )
                self.progress_callback(progress)

            # 世代ループ
            for generation in range(1, config.generations + 1):
                self.current_generation = generation

                logger.info(f"世代 {generation}/{config.generations} 開始")

                # 選択
                offspring = self.toolbox.select(population, len(population))
                offspring = list(map(self.toolbox.clone, offspring))

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

                # 無効な個体の評価
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # エリート保存
                population = self._apply_elitism(
                    population, offspring, config.elite_size
                )

                # 統計情報の記録
                record = self.stats.compile(population)
                self.logbook.record(gen=generation, evals=len(invalid_ind), **record)

                # 進捗コールバック
                if self.progress_callback:
                    progress = self._create_progress_info(
                        config, population, backtest_config.get("experiment_id", "")
                    )
                    self.progress_callback(progress)

                logger.info(f"世代 {generation} 完了 - 最高適応度: {record['max']:.4f}")

            # 結果の整理
            best_individual = tools.selBest(population, 1)[0]
            best_gene = decode_list_to_gene(best_individual)

            execution_time = time.time() - self.start_time

            result = {
                "best_strategy": best_gene,
                "best_fitness": best_individual.fitness.values[0],
                "population": population,
                "logbook": self.logbook,
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

    def _evaluate_individual(
        self,
        individual: List[float],
        config: GAConfig,
        backtest_config: Dict[str, Any] = None,
    ) -> Tuple[float]:
        """
        個体の評価（フィットネス計算）

        Args:
            individual: 評価する個体（数値リスト）
            config: GA設定
            backtest_config: バックテスト設定

        Returns:
            フィットネス値のタプル
        """
        try:
            logger.info(f"個体評価開始: 遺伝子長={len(individual)}")
            logger.info(f"バックテスト設定: {backtest_config}")
            # 数値リストから戦略遺伝子にデコード
            gene = decode_list_to_gene(individual)

            # 戦略の妥当性チェック
            is_valid, errors = self.strategy_factory.validate_gene(gene)
            if not is_valid:
                logger.debug(f"無効な戦略: {errors}")
                return (0.0,)  # 無効な戦略には最低スコア

            # 戦略クラスを生成
            self.strategy_factory.create_strategy_class(gene)

            # バックテスト設定を構築（ランダム時間足を使用）
            if backtest_config:
                logger.info(
                    f"バックテスト設定あり: {backtest_config.get('timeframe', 'N/A')}"
                )
                # ランダムな時間足設定を取得
                random_config = self._select_random_timeframe_config(backtest_config)
                logger.info(f"ランダム設定後: {random_config.get('timeframe', 'N/A')}")

                test_config = {
                    "strategy_name": f"GA_Generated_{gene.id}",
                    "symbol": random_config.get("symbol", "BTC/USDT"),
                    "timeframe": random_config.get("timeframe", "1d"),
                    "start_date": random_config.get("start_date", "2024-01-01"),
                    "end_date": random_config.get("end_date", "2024-04-09"),
                    "initial_capital": random_config.get("initial_capital", 100000),
                    "commission_rate": random_config.get("commission_rate", 0.001),
                    "strategy_config": {
                        "strategy_type": "GENERATED_TEST",
                        "parameters": {"strategy_gene": gene.to_dict()},
                    },
                }
            else:
                # フォールバック設定
                test_config = {
                    "strategy_name": f"GA_Generated_{gene.id}",
                    "symbol": "BTC/USDT",
                    "timeframe": "1d",
                    "start_date": "2024-01-01",
                    "end_date": "2024-04-09",
                    "initial_capital": 100000,
                    "commission_rate": 0.001,
                    "strategy_config": {
                        "strategy_type": "GENERATED_TEST",
                        "parameters": {"strategy_gene": gene.to_dict()},
                    },
                }

            # バックテスト実行
            result = self.backtest_service.run_backtest(test_config)

            # フィットネス計算
            fitness = self._calculate_fitness(result, config)

            return (fitness,)

        except Exception as e:
            logger.error(f"個体評価エラー: {e}")
            return (0.0,)  # エラー時は最低スコア

    def _evaluate_individual_wrapper(
        self, individual: List[float], config: GAConfig
    ) -> Tuple[float]:
        """評価関数のラッパー（バックテスト設定を渡す）"""
        logger.info(f"評価ラッパー呼び出し: {len(individual)}個の遺伝子")
        return self._evaluate_individual(
            individual, config, self._current_backtest_config
        )

    def _calculate_fitness(
        self, backtest_result: Dict[str, Any], config: GAConfig
    ) -> float:
        """
        バックテスト結果からフィットネスを計算

        Args:
            backtest_result: バックテスト結果
            config: GA設定

        Returns:
            フィットネス値
        """
        try:
            metrics = backtest_result.get("performance_metrics", {})

            # 制約条件のチェック（緩和版）
            constraints = config.fitness_constraints

            # 最小取引数チェック（緩和: 1取引以上）
            if metrics.get("total_trades", 0) < constraints.get("min_trades", 1):
                return 0.0

            # 最大ドローダウンチェック（緩和: 50%まで許可）
            max_drawdown = abs(metrics.get("max_drawdown", 1.0))
            if max_drawdown > constraints.get("max_drawdown_limit", 0.5):
                return 0.0

            # 最小シャープレシオチェック（緩和: -1.0まで許可）
            sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
            if sharpe_ratio < constraints.get("min_sharpe_ratio", -1.0):
                return 0.0

            # 強化されたフィットネス計算（GA真の目的に特化）
            fitness = 0.0
            weights = config.fitness_weights

            # 主要指標の取得
            total_return = metrics.get("total_return", 0.0)
            win_rate = (
                metrics.get("win_rate", 0.0) / 100.0
            )  # パーセンテージを小数に変換

            # 正規化（より実用的な範囲設定）
            # 1. リターン正規化: -50%〜+200% → 0〜1
            normalized_return = max(0, min(1, (total_return + 50) / 250))

            # 2. シャープレシオ正規化: -2〜+4 → 0〜1 (優秀な戦略は2以上)
            normalized_sharpe = max(0, min(1, (sharpe_ratio + 2) / 6))

            # 3. ドローダウン正規化: 0〜50% → 1〜0 (低いほど良い)
            normalized_drawdown = max(0, min(1, 1 - (max_drawdown / 0.5)))

            # 4. 勝率正規化: 0〜100% → 0〜1
            normalized_win_rate = max(0, min(1, win_rate))

            # 重み付き合計（GA真の目的に重点）
            fitness = (
                weights.get("total_return", 0.35) * normalized_return  # リターン重視
                + weights.get("sharpe_ratio", 0.35)
                * normalized_sharpe  # シャープレシオ重視
                + weights.get("max_drawdown", 0.25)
                * normalized_drawdown  # ドローダウン重視
                + weights.get("win_rate", 0.05) * normalized_win_rate  # 勝率は参考程度
            )

            # ボーナス: 優秀な戦略への追加評価
            if total_return > 20 and sharpe_ratio > 1.5 and max_drawdown < 0.15:
                fitness *= 1.2  # 20%ボーナス
            elif total_return > 50 and sharpe_ratio > 2.0 and max_drawdown < 0.10:
                fitness *= 1.5  # 50%ボーナス（非常に優秀）

            return fitness

        except Exception as e:
            logger.error(f"フィットネス計算エラー: {e}")
            return 0.0

    def _apply_constraints(self, func):
        """制約条件を適用するデコレータ"""

        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            # TODO: 制約条件の実装（パラメータ範囲チェック等）
            return result

        return wrapper

    def _apply_elitism(
        self, population: List, offspring: List, elite_size: int
    ) -> List:
        """エリート保存戦略を適用"""
        # 親世代と子世代を結合
        combined = population + offspring

        # フィットネスでソート
        combined.sort(key=lambda x: x.fitness.values[0], reverse=True)

        # 上位個体を選択
        return combined[: len(population)]

    def _create_progress_info(
        self, config: GAConfig, population: List, experiment_id: str
    ) -> GAProgress:
        """進捗情報を作成"""
        fitnesses = [ind.fitness.values[0] for ind in population if ind.fitness.valid]

        best_fitness = max(fitnesses) if fitnesses else 0.0
        avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0

        execution_time = time.time() - self.start_time
        estimated_remaining = (execution_time / max(1, self.current_generation)) * (
            config.generations - self.current_generation
        )

        return GAProgress(
            experiment_id=experiment_id,
            current_generation=self.current_generation,
            total_generations=config.generations,
            best_fitness=best_fitness,
            average_fitness=avg_fitness,
            execution_time=execution_time,
            estimated_remaining_time=estimated_remaining,
        )

    def set_progress_callback(self, callback: Callable[[GAProgress], None]):
        """進捗コールバックを設定"""
        self.progress_callback = callback

    def stop_evolution(self):
        """進化を停止"""
        self.is_running = False
        logger.info("進化停止が要求されました")
