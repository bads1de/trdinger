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
        self.toolbox: Optional[base.Toolbox] = None
        self.progress_callback: Optional[Callable] = None

        # 新しいランダム遺伝子生成器
        self.gene_generator = RandomGeneGenerator()

        # 統計情報
        self.stats: Optional[tools.Statistics] = None
        self.logbook: Optional[tools.Logbook] = None

        # 実行状態
        self.is_running = False
        self.current_generation = 0
        self.start_time = 0

        # 利用可能な時間足（動的に取得）
        self.available_timeframes = self._get_available_timeframes()

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
            creator.create("Individual", list, fitness=creator.FitnessMax)  # type: ignore

        # ツールボックスの初期化
        self.toolbox = base.Toolbox()

        # 遺伝子長の計算（v1仕様: 5指標×2 + エントリー条件3 + イグジット条件3 = 16）
        # gene_length = config.max_indicators * 2 + 6  # 現在未使用

        # 個体生成関数（新しいランダム遺伝子生成器を使用）
        def create_individual():
            """新しいランダム遺伝子生成器を使用して個体を生成"""
            try:
                # ランダム遺伝子生成器で戦略遺伝子を生成
                gene = self.gene_generator.generate_random_gene()

                # 戦略遺伝子を数値リストにエンコード
                individual = encode_gene_to_list(gene)

                return creator.Individual(individual)  # type: ignore
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

                return creator.Individual(individual)  # type: ignore

        self.toolbox.register("individual", create_individual)
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual  # type: ignore
        )

        # 評価関数（バックテスト設定は run_evolution で設定済み）
        # self._current_backtest_config = None
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
        self.logbook.header = "gen", "evals", "std", "min", "avg", "max"  # type: ignore

        logger.info("DEAP環境のセットアップ完了")

    def _get_available_timeframes(self) -> List[str]:
        """
        データベースから利用可能な時間軸を取得

        Returns:
            利用可能な時間軸のリスト
        """
        try:
            from database.connection import SessionLocal
            from database.repositories.ohlcv_repository import OHLCVRepository

            db = SessionLocal()
            try:
                repo = OHLCVRepository(db)
                symbols = repo.get_available_symbols()

                # BTC/USDT系のシンボルを優先的に使用
                target_symbols = ["BTC/USDT:USDT", "BTC/USDT", "BTCUSDT"]
                available_timeframes = []

                for symbol in target_symbols:
                    if symbol in symbols:
                        timeframes = repo.get_available_timeframes(symbol)
                        if timeframes:
                            available_timeframes = timeframes
                            logger.info(
                                f"利用可能な時間軸を取得: {symbol} -> {timeframes}"
                            )
                            break

                if not available_timeframes:
                    # フォールバック: 最初に見つかったシンボルの時間軸を使用
                    if symbols:
                        first_symbol = symbols[0]
                        available_timeframes = repo.get_available_timeframes(
                            first_symbol
                        )
                        logger.info(
                            f"フォールバック時間軸を取得: {first_symbol} -> {available_timeframes}"
                        )

                if not available_timeframes:
                    # 最終フォールバック
                    available_timeframes = ["1d"]
                    logger.warning(
                        "データベースに時間軸データが見つからないため、デフォルト値を使用"
                    )

                return available_timeframes

            finally:
                db.close()

        except Exception as e:
            logger.error(f"利用可能な時間軸の取得エラー: {e}")
            # エラー時のフォールバック
            return ["1d"]

    def _select_random_timeframe_config(
        self, base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        利用可能なデータに基づいてランダムな時間足を選択し、適切な期間を設定

        Args:
            base_config: ベースとなるバックテスト設定

        Returns:
            時間足と期間が調整された設定
        """
        try:
            from database.connection import SessionLocal
            from database.repositories.ohlcv_repository import OHLCVRepository
            from app.config.market_config import MarketDataConfig

            db = SessionLocal()
            try:
                repo = OHLCVRepository(db)
                symbols = repo.get_available_symbols()

                # シンボルの正規化
                input_symbol = base_config.get("symbol", "BTC/USDT")
                try:
                    # MarketDataConfigを使用してシンボルを正規化
                    if input_symbol == "BTC/USDT":
                        normalized_symbol = "BTC/USDT:USDT"  # データベース形式に変換
                    else:
                        normalized_symbol = MarketDataConfig.normalize_symbol(
                            input_symbol
                        )
                except ValueError:
                    # 正規化に失敗した場合のフォールバック
                    normalized_symbol = "BTC/USDT:USDT"
                    logger.warning(
                        f"シンボル正規化失敗、フォールバック使用: {input_symbol} -> {normalized_symbol}"
                    )

                # BTC/USDT系のシンボルを優先的に使用
                target_symbols = ["BTC/USDT:USDT", "BTC/USDT", "BTCUSDT"]
                selected_symbol = normalized_symbol
                available_timeframes = []

                # 正規化されたシンボルが利用可能かチェック
                if selected_symbol in symbols:
                    available_timeframes = repo.get_available_timeframes(
                        selected_symbol
                    )
                    logger.info(f"正規化シンボル使用: {selected_symbol}")

                # 正規化シンボルが利用できない場合、優先シンボルを使用
                if not available_timeframes:
                    for symbol in target_symbols:
                        if symbol in symbols:
                            timeframes = repo.get_available_timeframes(symbol)
                            if timeframes:
                                selected_symbol = symbol
                                available_timeframes = timeframes
                                logger.info(f"シンボル変更: {input_symbol} -> {symbol}")
                                break

                # それでも見つからない場合、最初のシンボルを使用
                if not available_timeframes and symbols:
                    selected_symbol = symbols[0]
                    available_timeframes = repo.get_available_timeframes(
                        selected_symbol
                    )
                    logger.info(f"フォールバックシンボル使用: {selected_symbol}")

                # 時間軸をランダム選択
                if available_timeframes:
                    selected_timeframe = random.choice(available_timeframes)
                    logger.info(f"ランダム時間足選択: {selected_timeframe}")
                else:
                    selected_timeframe = "1d"  # 最終フォールバック
                    logger.warning("利用可能な時間足が見つからず、デフォルト使用: 1d")

                logger.info(
                    f"選択されたシンボル・時間軸: {selected_symbol} {selected_timeframe}"
                )

            finally:
                db.close()

        except Exception as e:
            logger.error(f"データベース確認エラー: {e}")
            # エラー時のフォールバック
            selected_symbol = "BTC/USDT:USDT"
            selected_timeframe = "1d"

        # 元の設定の日付範囲を使用（現在時刻ベースではなく）
        original_start = base_config.get("start_date")
        original_end = base_config.get("end_date")

        if original_start and original_end:
            # 元の設定に日付がある場合はそれを使用
            start_date = original_start
            end_date = original_end
            logger.info(f"元の日付範囲を使用: {start_date} ～ {end_date}")
        else:
            # 日付が指定されていない場合のみ、現在時刻を基準に設定
            end_date_dt = datetime.now(timezone.utc)

            # 時間足に応じて適切な期間を設定
            if selected_timeframe == "15m":
                start_date_dt = end_date_dt - timedelta(days=7)  # 15分足: 1週間
            elif selected_timeframe == "30m":
                start_date_dt = end_date_dt - timedelta(days=14)  # 30分足: 2週間
            elif selected_timeframe == "1h":
                start_date_dt = end_date_dt - timedelta(days=30)  # 1時間足: 1ヶ月
            elif selected_timeframe == "4h":
                start_date_dt = end_date_dt - timedelta(days=60)  # 4時間足: 2ヶ月
            else:  # 1d
                start_date_dt = end_date_dt - timedelta(days=90)  # 日足: 3ヶ月

            start_date = start_date_dt.isoformat()
            end_date = end_date_dt.isoformat()
            logger.info(f"自動生成日付範囲: {start_date} ～ {end_date}")

        # 設定をコピーして更新
        config = base_config.copy()
        config["symbol"] = selected_symbol
        config["timeframe"] = selected_timeframe
        config["start_date"] = start_date
        config["end_date"] = end_date

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

            # 評価環境固定化: GA実行開始時に一度だけバックテスト設定を決定
            if backtest_config:
                logger.info("評価環境を固定化中...")
                self._fixed_backtest_config = self._select_random_timeframe_config(
                    backtest_config
                )
                logger.info(f"固定化された評価環境: {self._fixed_backtest_config}")
            else:
                self._fixed_backtest_config = None
                logger.info(
                    "バックテスト設定が提供されていないため、フォールバック設定を使用"
                )

            # DEAP環境のセットアップ
            self.setup_deap(config)

            # 初期個体群の生成
            if self.toolbox is None:
                raise RuntimeError(
                    "DEAP環境がセットアップされていません。setup_deap()を先に実行してください。"
                )
            population = self.toolbox.population(n=config.population_size)  # type: ignore

            # 初期評価
            logger.info("初期個体群の評価開始...")
            fitnesses = self.toolbox.map(self.toolbox.evaluate, population)  # type: ignore
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # 統計情報の記録
            if self.stats is None or self.logbook is None:
                raise RuntimeError(
                    "統計情報が初期化されていません。setup_deap()を先に実行してください。"
                )
            record = self.stats.compile(population)  # type: ignore
            self.logbook.record(gen=0, evals=len(population), **record)  # type: ignore

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
                offspring = self.toolbox.select(population, len(population))  # type: ignore
                offspring = list(map(self.toolbox.clone, offspring))  # type: ignore

                # 交叉
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < config.crossover_rate:
                        self.toolbox.mate(child1, child2)  # type: ignore
                        del child1.fitness.values
                        del child2.fitness.values

                # 突然変異
                for mutant in offspring:
                    if random.random() < config.mutation_rate:
                        self.toolbox.mutate(mutant)  # type: ignore
                        del mutant.fitness.values

                # 無効な個体の評価
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)  # type: ignore
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # エリート保存
                population = self._apply_elitism(
                    population, offspring, config.elite_size
                )

                # 統計情報の記録
                record = self.stats.compile(population)  # type: ignore
                self.logbook.record(gen=generation, evals=len(invalid_ind), **record)  # type: ignore

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
        backtest_config: Optional[Dict[str, Any]] = None,
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
            # ログレベルに応じた出力制御
            if config.enable_detailed_logging:
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

            # バックテスト設定を構築（固定化された設定を使用）
            if hasattr(self, "_fixed_backtest_config") and self._fixed_backtest_config:
                if config.enable_detailed_logging:
                    logger.debug(
                        f"固定化された設定を使用: {self._fixed_backtest_config.get('timeframe', 'N/A')}"
                    )
                fixed_config = self._fixed_backtest_config

                test_config = {
                    "strategy_name": f"GA_Generated_{gene.id}",
                    "symbol": fixed_config.get("symbol", "BTC/USDT"),
                    "timeframe": fixed_config.get("timeframe", "1d"),
                    "start_date": fixed_config.get("start_date", "2024-01-01"),
                    "end_date": fixed_config.get("end_date", "2024-04-09"),
                    "initial_capital": fixed_config.get("initial_capital", 100000),
                    "commission_rate": fixed_config.get("commission_rate", 0.001),
                    "strategy_config": {
                        "strategy_type": "GENERATED_TEST",
                        "parameters": {"strategy_gene": gene.to_dict()},
                    },
                }
            elif backtest_config:
                if config.enable_detailed_logging:
                    logger.debug(
                        f"フォールバック: 提供された設定を使用: {backtest_config.get('timeframe', 'N/A')}"
                    )
                test_config = {
                    "strategy_name": f"GA_Generated_{gene.id}",
                    "symbol": backtest_config.get("symbol", "BTC/USDT"),
                    "timeframe": backtest_config.get("timeframe", "1d"),
                    "start_date": backtest_config.get("start_date", "2024-01-01"),
                    "end_date": backtest_config.get("end_date", "2024-04-09"),
                    "initial_capital": backtest_config.get("initial_capital", 100000),
                    "commission_rate": backtest_config.get("commission_rate", 0.001),
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
        self,
        population: List,
        offspring: List,
        elite_size: int,
    ) -> List:
        """エリート保存戦略を適用"""
        # 親世代から上位elite_size個体を保存
        population.sort(key=lambda x: x.fitness.values[0], reverse=True)
        elite = population[:elite_size]

        # 子世代から残りの個体を選択
        offspring.sort(key=lambda x: x.fitness.values[0], reverse=True)
        remaining_size = len(population) - elite_size
        selected_offspring = offspring[:remaining_size]

        # エリートと選択された子世代を結合
        return elite + selected_offspring

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
