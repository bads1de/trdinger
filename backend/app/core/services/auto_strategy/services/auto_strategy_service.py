"""
自動戦略生成サービス

GA実行、進捗管理、結果保存などの統合機能を提供します。
"""

import uuid
import time
import threading
from typing import Dict, Any, List, Optional, Callable, cast
import logging

from ..models.strategy_gene import StrategyGene
from ..models.ga_config import GAConfig, GAProgress
from ..engines.ga_engine import GeneticAlgorithmEngine
from ..factories.strategy_factory import StrategyFactory
from app.core.services.backtest_service import BacktestService
from app.core.services.backtest_data_service import BacktestDataService
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.ga_experiment_repository import GAExperimentRepository
from database.repositories.generated_strategy_repository import (
    GeneratedStrategyRepository,
)
from database.repositories.backtest_result_repository import BacktestResultRepository
from database.connection import SessionLocal

logger = logging.getLogger(__name__)


class AutoStrategyService:
    """
    自動戦略生成サービス

    GA実行、進捗管理、結果保存を統合的に管理します。
    """

    def __init__(self):
        """初期化"""
        self.running_experiments: Dict[str, Dict[str, Any]] = (
            {}
        )  # 実行中のGA実験を管理する辞書
        self.progress_data: Dict[str, GAProgress] = (
            {}
        )  # 各実験の進捗データを保持する辞書

        # サービスの初期化
        self.strategy_factory = StrategyFactory()
        self.backtest_service: BacktestService
        self.ga_engine: GeneticAlgorithmEngine

        # リポジトリの初期化
        self.ga_experiment_repo = None  # GA実験の永続化を担当するリポジトリ
        self.generated_strategy_repo = (
            None  # 生成された戦略の永続化を担当するリポジトリ
        )

        self._init_services()

    def _init_services(self):
        """
        サービスの初期化を行います。

        このメソッドは、AutoStrategyService が依存する他のサービスやリポジトリを
        初期化するために呼び出されます。特に、データベース接続を確立し、
        OHLCV、Open Interest、Funding Rate などのデータリポジトリ、
        GA実験と生成戦略のリポジトリ、そしてバックテストサービスとGAエンジンを
        設定します。
        """
        try:
            # データベースセッションの開始
            db = SessionLocal()
            try:
                # 各種データリポジトリの初期化
                ohlcv_repo = OHLCVRepository(
                    db
                )  # OHLCV (始値、高値、安値、終値、出来高) データのリポジトリ
                oi_repo = OpenInterestRepository(
                    db
                )  # Open Interest (建玉) データのリポジトリ
                fr_repo = FundingRateRepository(
                    db
                )  # Funding Rate (資金調達率) データのリポジトリ
                self.ga_experiment_repo = GAExperimentRepository(
                    db
                )  # GA実験結果を保存・取得するリポジトリ
                self.generated_strategy_repo = GeneratedStrategyRepository(
                    db
                )  # GAによって生成された戦略を保存・取得するリポジトリ

                # バックテストデータサービスを初期化 (OHLCV, OI, FR データを統合)
                data_service = BacktestDataService(
                    ohlcv_repo=ohlcv_repo, oi_repo=oi_repo, fr_repo=fr_repo
                )
                # バックテストサービスを初期化 (データサービスを利用)
                self.backtest_service = BacktestService(data_service)

                # 遺伝的アルゴリズムエンジンを初期化 (バックテストサービスと戦略ファクトリを利用)
                self.ga_engine = GeneticAlgorithmEngine(
                    self.backtest_service, self.strategy_factory
                )

                logger.info("自動戦略生成サービス初期化完了（OI/FR統合版）")

            finally:
                db.close()

        except Exception as e:
            logger.error(f"AutoStrategyServiceの初期化中にエラーが発生しました: {e}")
            raise

    def start_strategy_generation(
        self,
        experiment_name: str,
        ga_config: GAConfig,
        backtest_config: Dict[str, Any],
        progress_callback: Optional[Callable[[GAProgress], None]] = None,
    ) -> str:
        """
        戦略生成を開始

        Args:
            experiment_name: 実験名
            ga_config: GA設定
            backtest_config: バックテスト設定
            progress_callback: 進捗コールバック

        Returns:
            実験ID
        """
        try:
            logger.info("=== 戦略生成開始処理開始 ===")
            logger.info(f"実験名: {experiment_name}")

            # サービスが正しく初期化されているか依存関係を確認
            logger.info("依存関係の確認中...")
            if self.ga_engine is None:  # GAエンジンが初期化されているか
                raise RuntimeError("GAエンジンが初期化されていません。")
            if (
                self.backtest_service is None
            ):  # バックテストサービスが初期化されているか
                raise RuntimeError("バックテストサービスが初期化されていません。")
            if self.ga_experiment_repo is None:  # GA実験リポジトリが初期化されているか
                raise RuntimeError("GA実験リポジトリが初期化されていません。")
            logger.info("依存関係の確認完了")

            # 新しいGA実験のための一意なIDを生成
            logger.info("実験IDを生成中...")
            experiment_id = str(
                uuid.uuid4()
            )  # UUID (Universally Unique Identifier) を使用して一意なIDを生成
            logger.info(f"実験ID生成完了: {experiment_id}")

            # GA設定オブジェクトの妥当性を検証
            logger.info("GA設定の検証中...")
            is_valid, errors = (
                ga_config.validate()
            )  # GAConfig クラスの validate メソッドを呼び出し
            if not is_valid:  # 検証に失敗した場合
                logger.error(
                    f"GA設定の検証に失敗しました: {errors}"
                )  # エラーログを出力
                raise ValueError(
                    f"無効なGA設定です: {', '.join(errors)}"
                )  # ValueError を発生させる
            logger.info("GA設定の検証が完了しました。")

            # 新しいGA実験の情報をデータベースに永続化
            logger.info("データベースに実験を保存中...")
            db = SessionLocal()  # データベースセッションを取得
            try:
                logger.info("データベースセッション作成完了")
                ga_experiment_repo = GAExperimentRepository(
                    db
                )  # GAExperimentRepository のインスタンスを作成
                logger.info("GA実験リポジトリ作成完了")

                db_experiment = ga_experiment_repo.create_experiment(
                    name=experiment_name,
                    config=ga_config.to_dict(),
                    total_generations=ga_config.generations,
                    status="starting",
                )
                db_experiment_id = cast(int, db_experiment.id)
                logger.info(f"データベース実験作成完了: DB ID = {db_experiment_id}")
            except Exception as db_error:
                logger.error(
                    f"データベース操作中にエラーが発生しました: {type(db_error).__name__}: {str(db_error)}"
                )
                raise
            finally:
                db.close()
                logger.info("データベースセッション終了")

            # 実行中の実験情報をメモリ上に記録
            logger.info("実験情報を記録中...")
            experiment_info = {
                "id": experiment_id,  # 生成された実験ID
                "db_id": db_experiment_id,  # データベースに保存された実験のID
                "name": experiment_name,  # 実験名
                "ga_config": ga_config,  # GA設定オブジェクト
                "backtest_config": backtest_config,  # バックテスト設定辞書
                "status": "starting",  # 実験の現在のステータス
                "start_time": time.time(),  # 実験開始時刻のタイムスタンプ
                "thread": None,  # 実験を実行するスレッドオブジェクト (後で設定)
            }
            logger.info("実験情報記録完了")

            self.running_experiments[experiment_id] = experiment_info

            # GAエンジンからの進捗更新を受け取るコールバック関数を設定
            logger.info("進捗コールバックを設定中...")

            def combined_callback(progress: GAProgress):
                # メモリ上の進捗データを更新
                self.progress_data[experiment_id] = progress

                # データベースに進捗を保存
                try:
                    db = SessionLocal()
                    try:
                        ga_experiment_repo = GAExperimentRepository(db)
                        ga_experiment_repo.update_experiment_progress(
                            experiment_id=db_experiment_id,
                            current_generation=progress.current_generation,
                            progress=progress.progress_percentage / 100.0,
                            best_fitness=progress.best_fitness,
                        )
                    finally:
                        db.close()
                except Exception as e:
                    logger.error(
                        f"GA進捗のデータベース保存中にエラーが発生しました: {e}"
                    )

                if progress_callback:
                    progress_callback(progress)

            logger.info("進捗コールバックを設定中...")
            self.ga_engine.set_progress_callback(combined_callback)
            logger.info("進捗コールバック設定完了")

            # GAの実行は時間がかかるため、バックグラウンドスレッドで非同期に実行します。
            # これにより、GA実行中もAPIサーバーは他のリクエストを処理できます。
            logger.info("バックグラウンドスレッドを作成中...")
            thread = threading.Thread(
                target=self._run_experiment,  # スレッドで実行する関数
                args=(
                    experiment_id,
                    ga_config,
                    backtest_config,
                ),  # _run_experiment に渡す引数
                daemon=True,  # メインスレッドが終了すると、このスレッドも自動的に終了します。
            )
            logger.info("スレッド作成完了")

            experiment_info["thread"] = thread
            experiment_info["status"] = "running"

            logger.info("スレッドを開始中...")
            thread.start()
            logger.info("スレッド開始完了")

            logger.info(f"戦略生成実験開始成功: {experiment_id} ({experiment_name})")
            return experiment_id

        except Exception as e:
            import traceback

            error_msg = str(e)
            error_type = type(e).__name__
            traceback_str = traceback.format_exc()

            logger.error(f"戦略生成開始エラー - 例外型: {error_type}")
            logger.error(f"戦略生成開始エラー - メッセージ: {error_msg}")
            logger.error(f"戦略生成開始エラー - トレースバック:\n{traceback_str}")

            # エラーメッセージが空の場合の対処
            if not error_msg:
                error_msg = f"不明な {error_type} エラーが発生しました"

            # 詳細なエラー情報を含む例外を再発生
            detailed_error = f"{error_type}: {error_msg}"
            raise RuntimeError(detailed_error) from e

    def _run_experiment(
        self, experiment_id: str, ga_config: GAConfig, backtest_config: Dict[str, Any]
    ):
        """
        実験をバックグラウンドで実行

        Args:
            experiment_id: 実験ID
            ga_config: GA設定
            backtest_config: バックテスト設定
        """
        try:
            experiment_info = self.running_experiments[experiment_id]

            # バックテスト設定に実験IDを追加
            backtest_config["experiment_id"] = experiment_id

            # GeneticAlgorithmEngine を使用してGAを実行
            logger.info(f"GA実行開始: {experiment_id}")
            result = self.ga_engine.run_evolution(
                ga_config, backtest_config
            )  # GAの進化プロセスを開始

            # GA実行結果をデータベースに保存
            self._save_experiment_result(
                experiment_id, result, ga_config, backtest_config
            )

            # 実験のステータスを「完了」に更新
            experiment_info["status"] = "completed"
            experiment_info["end_time"] = time.time()  # 終了時刻を記録
            experiment_info["result"] = result  # 最終結果を保存

            # データベース上の実験レコードを「完了」状態に更新
            try:
                db = SessionLocal()  # データベースセッションを取得
                try:
                    ga_experiment_repo = GAExperimentRepository(
                        db
                    )  # GAExperimentRepository のインスタンスを作成
                    ga_experiment_repo.complete_experiment(  # 実験を完了状態に更新
                        experiment_id=experiment_info[
                            "db_id"
                        ],  # データベース上の実験ID
                        best_fitness=result["best_fitness"],  # 最良の適応度
                        final_generation=ga_config.generations,  # 完了した世代数
                    )
                finally:
                    db.close()  # データベースセッションを閉じる
            except Exception as e:
                logger.error(
                    f"実験完了状態のデータベース更新中にエラーが発生しました: {e}"
                )

            # 最終的な進捗情報をGAProgressオブジェクトとして作成
            final_progress = GAProgress(
                experiment_id=experiment_id,  # 実験ID
                current_generation=ga_config.generations,  # 完了した世代数
                total_generations=ga_config.generations,  # 総世代数
                best_fitness=result["best_fitness"],  # 最終的な最良適応度
                average_fitness=result[
                    "best_fitness"
                ],  # 簡略化のため、最良適応度を平均適応度として設定
                execution_time=result["execution_time"],  # 総実行時間
                estimated_remaining_time=0.0,  # 残り推定時間は0
                status="completed",  # ステータスを「完了」に設定
            )

            self.progress_data[experiment_id] = final_progress

            logger.info(f"GA実行完了: {experiment_id}")

        except Exception as e:
            logger.error(f"GA実験の実行中にエラーが発生しました ({experiment_id}): {e}")

            # エラーが発生した場合、実験の状態を「エラー」として記録
            if (
                experiment_id in self.running_experiments
            ):  # 実行中の実験リストに存在する場合
                self.running_experiments[experiment_id][
                    "status"
                ] = "error"  # ステータスを「エラー」に設定
                self.running_experiments[experiment_id]["error"] = str(
                    e
                )  # エラーメッセージを記録

                # データベース上の実験レコードを「エラー」状態に更新
                try:
                    db = SessionLocal()  # データベースセッションを取得
                    try:
                        ga_experiment_repo = GAExperimentRepository(
                            db
                        )  # GAExperimentRepository のインスタンスを作成
                        ga_experiment_repo.update_experiment_status(  # 実験のステータスを更新
                            experiment_id=self.running_experiments[experiment_id][
                                "db_id"
                            ],  # データベース上の実験ID
                            status="error",  # ステータスを「エラー」に設定
                        )
                    finally:
                        db.close()  # データベースセッションを閉じる
                except Exception as db_error:
                    logger.error(
                        f"エラー状態のデータベース更新中にエラーが発生しました: {db_error}"
                    )

            # エラー発生時の進捗情報をGAProgressオブジェクトとして作成
            error_progress = GAProgress(
                experiment_id=experiment_id,  # 実験ID
                current_generation=0,  # 世代数は0 (エラーのため)
                total_generations=ga_config.generations,  # 総世代数
                best_fitness=0.0,  # 最良適応度は0.0 (エラーのため)
                average_fitness=0.0,  # 平均適応度も0.0
                execution_time=0.0,  # 実行時間も0.0
                estimated_remaining_time=0.0,  # 残り推定時間も0.0
                status="error",  # ステータスを「エラー」に設定
            )

            self.progress_data[experiment_id] = error_progress

    def get_experiment_progress(self, experiment_id: str) -> Optional[GAProgress]:
        """
        実験の進捗を取得

        Args:
            experiment_id: 実験ID

        Returns:
            進捗情報（存在しない場合はNone）
        """
        return self.progress_data.get(experiment_id)

    def get_experiment_result(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        実験結果を取得

        Args:
            experiment_id: 実験ID

        Returns:
            実験結果（存在しない場合はNone）
        """
        experiment_info = self.running_experiments.get(experiment_id)
        if experiment_info and experiment_info["status"] == "completed":
            return experiment_info.get("result")
        return None

    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        実験一覧を取得

        Returns:
            実験情報のリスト
        """
        experiments = []
        for experiment_id, info in self.running_experiments.items():
            experiment_summary = {
                "id": experiment_id,
                "name": info["name"],
                "status": info["status"],
                "start_time": info["start_time"],
                "end_time": info.get("end_time"),
                "error": info.get("error"),
            }
            experiments.append(experiment_summary)

        return experiments

    def stop_experiment(self, experiment_id: str) -> bool:
        """
        実験を停止

        Args:
            experiment_id: 実験ID

        Returns:
            停止成功の場合True
        """
        try:
            experiment_info = self.running_experiments.get(experiment_id)
            if not experiment_info:
                return False

            if experiment_info["status"] == "running":
                # GA実行を停止
                self.ga_engine.stop_evolution()

                # 実験状態を更新
                experiment_info["status"] = "stopped"
                experiment_info["end_time"] = time.time()

                logger.info(f"実験停止: {experiment_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"GA実験の停止中にエラーが発生しました: {e}")
            return False

    def _save_experiment_result(
        self,
        experiment_id: str,
        result: Dict[str, Any],
        ga_config: GAConfig,
        backtest_config: Dict[str, Any],
    ):
        """
        実験結果をデータベースに保存

        Args:
            experiment_id: 実験ID
            result: GA実行結果
            ga_config: GA設定
            backtest_config: バックテスト設定
        """
        try:
            experiment_info = self.running_experiments.get(experiment_id)
            if not experiment_info:
                logger.error(
                    f"指定された実験IDの実験情報が見つかりません: {experiment_id}"
                )
                return

            db_experiment_id = experiment_info["db_id"]
            best_strategy = result["best_strategy"]
            best_fitness = result["best_fitness"]
            all_strategies = result.get("all_strategies", [])

            logger.info(f"実験結果保存開始: {experiment_id}")
            logger.info(f"最高フィットネス: {best_fitness:.4f}")
            logger.info(f"最良戦略ID: {best_strategy.id}")
            logger.info(f"使用指標数: {len(best_strategy.indicators)}")
            logger.info(f"保存対象戦略数: {len(all_strategies)}")

            # データベースに戦略を保存
            db = SessionLocal()
            try:
                generated_strategy_repo = GeneratedStrategyRepository(db)
                backtest_result_repo = BacktestResultRepository(db)

                # GAによって見つかった最良戦略をデータベースに保存
                best_strategy_record = generated_strategy_repo.save_strategy(
                    experiment_id=db_experiment_id,  # 関連するGA実験のデータベースID
                    gene_data=best_strategy.to_dict(),  # 戦略遺伝子の辞書表現
                    generation=ga_config.generations,  # 戦略が生成された世代
                    fitness_score=best_fitness,  # 戦略の適応度スコア
                )

                logger.info(f"最良戦略を保存しました: DB ID {best_strategy_record.id}")

                # 最良戦略のバックテスト結果をbacktest_resultsテーブルにも保存
                try:
                    logger.info("最良戦略のバックテスト結果を詳細保存開始...")

                    # GAの評価フェーズでは高速化のために簡略化されたフィットネス計算を行いますが、
                    # ここでは最終的に見つかった最良戦略について、詳細なパフォーマンス指標や
                    # 取引履歴を含む完全なバックテストを再実行し、その結果を永続化します。
                    detailed_backtest_config = backtest_config.copy()
                    detailed_backtest_config["strategy_name"] = (
                        f"AUTO_STRATEGY_{experiment_info['name']}_{best_strategy.id[:8]}"
                    )
                    detailed_backtest_config["strategy_config"] = {
                        "strategy_type": "GENERATED_AUTO",
                        "parameters": {"strategy_gene": best_strategy.to_dict()},
                    }

                    detailed_result = self.backtest_service.run_backtest(
                        detailed_backtest_config
                    )

                    # backtest_results テーブルに保存するためのデータ構造を構築
                    backtest_result_data = {
                        "strategy_name": detailed_backtest_config[
                            "strategy_name"
                        ],  # 戦略名
                        "symbol": detailed_backtest_config["symbol"],  # シンボル
                        "timeframe": detailed_backtest_config["timeframe"],  # 時間足
                        "start_date": detailed_backtest_config[
                            "start_date"
                        ],  # バックテスト開始日
                        "end_date": detailed_backtest_config[
                            "end_date"
                        ],  # バックテスト終了日
                        "initial_capital": detailed_backtest_config[
                            "initial_capital"
                        ],  # 初期資金
                        "commission_rate": detailed_backtest_config.get(  # 手数料率 (デフォルト値あり)
                            "commission_rate", 0.001
                        ),
                        "config_json": {  # バックテスト設定のJSON形式
                            "strategy_config": detailed_backtest_config[
                                "strategy_config"
                            ],
                            "experiment_id": experiment_id,  # 実験ID
                            "db_experiment_id": db_experiment_id,  # データベース実験ID
                            "fitness_score": best_fitness,  # 適応度スコア
                        },
                        "performance_metrics": detailed_result.get(  # パフォーマンス指標 (デフォルト値あり)
                            "performance_metrics", {}
                        ),
                        "equity_curve": detailed_result.get(
                            "equity_curve", []
                        ),  # エクイティカーブデータ (デフォルト値あり)
                        "trade_history": detailed_result.get(
                            "trade_history", []
                        ),  # 取引履歴データ (デフォルト値あり)
                        "execution_time": detailed_result.get(
                            "execution_time"
                        ),  # 実行時間
                        "status": "completed",  # ステータスを「完了」に設定
                    }

                    # 構築したデータを backtest_results テーブルに保存
                    saved_backtest_result = backtest_result_repo.save_backtest_result(
                        backtest_result_data
                    )
                    logger.info(
                        f"最良戦略のバックテスト結果を保存しました: ID {saved_backtest_result.get('id')}"
                    )

                except Exception as e:
                    logger.error(
                        f"最良戦略のバックテスト結果の保存中にエラーが発生しました: {e}"
                    )
                    # エラーが発生してもメイン処理は継続

                # GAによって生成された全ての戦略を一括でデータベースに保存 (オプション機能)
                if (
                    all_strategies and len(all_strategies) > 1
                ):  # 全戦略が存在し、かつ複数ある場合
                    strategies_data = []  # 保存する戦略データのリスト
                    for i, strategy in enumerate(
                        all_strategies[
                            :100
                        ]  # 最大100戦略までを対象とする (パフォーマンス考慮)
                    ):
                        if (
                            strategy != best_strategy
                        ):  # 最良戦略は既に保存済みなのでスキップ
                            strategies_data.append(
                                {
                                    "experiment_id": db_experiment_id,  # 関連するGA実験のデータベースID
                                    "gene_data": strategy.to_dict(),  # 戦略遺伝子の辞書表現
                                    "generation": ga_config.generations,  # 戦略が生成された世代
                                    "fitness_score": result.get(  # 適応度スコア (デフォルト値あり)
                                        "fitness_scores", [0.0] * len(all_strategies)
                                    )[
                                        i
                                    ],
                                }
                            )

                    if strategies_data:
                        saved_strategies = (
                            generated_strategy_repo.save_strategies_batch(
                                strategies_data
                            )
                        )
                        logger.info(
                            f"追加戦略を一括保存しました: {len(saved_strategies)} 件"
                        )

            finally:
                db.close()

            logger.info(f"実験結果保存完了: {experiment_id}")

        except Exception as e:
            logger.error(f"GA実験結果の保存中にエラーが発生しました: {e}")
            raise

    def validate_strategy_gene(self, gene: StrategyGene) -> tuple[bool, List[str]]:
        """
        戦略遺伝子の妥当性を検証

        Args:
            gene: 検証する戦略遺伝子

        Returns:
            (is_valid, error_messages)
        """
        return self.strategy_factory.validate_gene(gene)

    def test_strategy_generation(
        self, gene: StrategyGene, backtest_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        単一戦略のテスト生成・実行

        Args:
            gene: テストする戦略遺伝子
            backtest_config: バックテスト設定

        Returns:
            テスト結果
        """
        try:
            # 戦略の妥当性チェック
            is_valid, errors = self.validate_strategy_gene(gene)
            if not is_valid:
                return {"success": False, "errors": errors}

            # 戦略クラスを生成
            self.strategy_factory.create_strategy_class(gene)

            # バックテスト実行
            test_config = backtest_config.copy()
            test_config["strategy_name"] = f"TEST_{gene.id}"
            test_config["strategy_config"] = {
                "strategy_type": "GENERATED_TEST",
                "parameters": {"strategy_gene": gene.to_dict()},
            }

            result = self.backtest_service.run_backtest(test_config)

            return {
                "success": True,
                "strategy_gene": gene.to_dict(),
                "backtest_result": result,
            }

        except Exception as e:
            logger.error(f"戦略のテスト実行中にエラーが発生しました: {e}")
            return {"success": False, "error": str(e)}
