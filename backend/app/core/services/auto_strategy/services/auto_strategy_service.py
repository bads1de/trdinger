"""
自動戦略生成サービス

GA実行、進捗管理、結果保存などの統合機能を提供します。
責任を分離し、各機能を専用モジュールに委譲します。
"""

import logging
from typing import Dict, Any, List, Optional, Callable

from ..models.ga_config import GAConfig, GAProgress
from ..models.strategy_gene import StrategyGene
from ..engines.ga_engine import GeneticAlgorithmEngine
from ..factories.strategy_factory import StrategyFactory
from app.core.services.backtest_service import BacktestService
from app.core.services.backtest_data_service import BacktestDataService
from .experiment_manager import ExperimentManager
from .progress_tracker import ProgressTracker

# データベースリポジトリのインポート
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.generated_strategy_repository import (
    GeneratedStrategyRepository,
)
from database.repositories.ga_experiment_repository import GAExperimentRepository
from database.repositories.backtest_result_repository import (
    BacktestResultRepository,
)
from database.connection import SessionLocal


logger = logging.getLogger(__name__)


class AutoStrategyService:
    """
    自動戦略生成サービス

    GA実行、進捗管理、結果保存を統合的に管理します。
    各機能は専用モジュールに委譲し、メインのAPI機能に集中します。
    """

    def __init__(self):
        """初期化"""
        # 分離されたコンポーネント
        self.experiment_manager = ExperimentManager()
        self.progress_tracker = ProgressTracker()

        # サービスの初期化
        self.strategy_factory = StrategyFactory()
        self.backtest_service: BacktestService
        self.ga_engine: GeneticAlgorithmEngine

        self._init_services()

    def _validate_dependencies(self):
        """依存関係の確認"""
        if self.ga_engine is None:
            raise RuntimeError("GAエンジンが初期化されていません。")
        if self.backtest_service is None:
            raise RuntimeError("バックテストサービスが初期化されていません。")

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
                # このサービスは、バックテストに必要な市場データ（OHLCV、建玉、資金調達率）を
                # データベースから取得し、整形する役割を担います。
                data_service = BacktestDataService(
                    ohlcv_repo=ohlcv_repo, oi_repo=oi_repo, fr_repo=fr_repo
                )
                # バックテストサービスを初期化 (データサービスを利用)
                # このサービスは、特定の戦略と市場データを用いてバックテストを実行し、
                # そのパフォーマンスを評価する役割を担います。
                self.backtest_service = BacktestService(data_service)

                # 遺伝的アルゴリズムエンジンを初期化 (バックテストサービスと戦略ファクトリを利用)
                # このエンジンは、遺伝的アルゴリズムの中核を実装し、戦略の生成、評価、選択、
                # 交叉、突然変異といった進化プロセスを管理します。
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
            logger.info(f"戦略生成開始: {experiment_name}")

            # 依存関係の確認
            self._validate_dependencies()

            # GA設定の検証
            is_valid, errors = ga_config.validate()
            if not is_valid:
                raise ValueError(f"無効なGA設定です: {', '.join(errors)}")

            # 実験を作成
            experiment_id = self.experiment_manager.create_experiment(
                experiment_name, ga_config, backtest_config
            )

            # 進捗コールバックを設定
            experiment_info = self.experiment_manager.get_experiment_info(experiment_id)
            if experiment_info:
                if progress_callback:
                    self.progress_tracker.set_progress_callback(
                        experiment_id, progress_callback, experiment_info["db_id"]
                    )

                # GAエンジンに進捗コールバックを設定
                ga_callback = self.progress_tracker.get_progress_callback(experiment_id)
                if ga_callback:
                    self.ga_engine.set_progress_callback(ga_callback)

            # 実験を開始
            success = self.experiment_manager.start_experiment(
                experiment_id, self._run_experiment
            )

            if not success:
                raise RuntimeError("実験の開始に失敗しました")

            logger.info(f"戦略生成実験開始成功: {experiment_id}")
            return experiment_id

        except Exception as e:
            logger.error(f"戦略生成開始エラー: {e}")
            raise

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
            # バックテスト設定に実験IDを追加
            backtest_config["experiment_id"] = experiment_id

            # GA実行
            logger.info(f"GA実行開始: {experiment_id}")
            result = self.ga_engine.run_evolution(ga_config, backtest_config)

            # 実験結果を保存
            self._save_experiment_result(
                experiment_id, result, ga_config, backtest_config
            )

            # 実験を完了状態にする
            self.experiment_manager.complete_experiment(experiment_id, result)

            # 最終進捗を作成・通知
            self.progress_tracker.create_final_progress(
                experiment_id, result, ga_config
            )

            logger.info(f"GA実行完了: {experiment_id}")

        except Exception as e:
            logger.error(f"GA実験の実行中にエラーが発生しました ({experiment_id}): {e}")

            # 実験を失敗状態にする
            self.experiment_manager.fail_experiment(experiment_id, str(e))

            # エラー進捗を作成・通知
            self.progress_tracker.create_error_progress(
                experiment_id, ga_config, str(e)
            )

    def get_experiment_progress(self, experiment_id: str) -> Optional[GAProgress]:
        """実験の進捗を取得"""
        return self.progress_tracker.get_progress(experiment_id)

    def get_experiment_result(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """実験結果を取得"""
        return self.experiment_manager.get_experiment_result(experiment_id)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """実験一覧を取得"""
        return self.experiment_manager.list_experiments()

    def stop_experiment(self, experiment_id: str) -> bool:
        """実験を停止"""
        try:
            # GA実行を停止
            self.ga_engine.stop_evolution()

            # 実験を停止状態にする
            success = self.experiment_manager.stop_experiment(experiment_id)

            if success:
                logger.info(f"実験停止: {experiment_id}")

            return success

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
            experiment_info = self.experiment_manager.get_experiment_info(experiment_id)
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
                # GAの評価フェーズでは高速化のために簡略化されたフィットネス計算を行いますが、
                # ここでは最終的に見つかった最良戦略について、詳細なパフォーマンス指標や
                # 取引履歴を含む完全なバックテストを再実行し、その結果を永続化します。
                try:
                    logger.info("最良戦略のバックテスト結果を詳細保存開始...")

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
                        "commission_rate": detailed_backtest_config.get(
                            "commission_rate", 0.001
                        ),  # 手数料率 (デフォルト値あり)
                        "config_json": {  # バックテスト設定のJSON形式
                            "strategy_config": detailed_backtest_config[
                                "strategy_config"
                            ],
                            "experiment_id": experiment_id,  # 実験ID
                            "db_experiment_id": db_experiment_id,  # データベース実験ID
                            "fitness_score": best_fitness,  # 適応度スコア
                        },
                        "performance_metrics": detailed_result.get(
                            "performance_metrics", {}
                        ),  # パフォーマンス指標 (デフォルト値あり)
                        # "equity_curve": detailed_result.get(
                        #     "equity_curve", []
                        # ),  # エクイティカーブデータ (デフォルト値あり)
                        # "trade_history": detailed_result.get(
                        #     "trade_history", []
                        # ),  # 取引履歴データ (デフォルト値あり)
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
                # 大量の戦略が生成される可能性があるため、パフォーマンスを考慮し、
                # 最良戦略以外の戦略もまとめて保存します。
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
