"""
オートストラテジー統合管理サービス

APIルーター内に散在していたオートストラテジー関連のビジネスロジックを統合管理します。
責務の分離とSOLID原則に基づいた設計を実現します。
"""

import logging
from typing import Any, Dict, List, Optional
from fastapi import BackgroundTasks

from app.services.auto_strategy.models.gene_strategy import StrategyGene
from app.services.auto_strategy.managers.experiment_manager import ExperimentManager
from app.services.backtest.backtest_data_service import BacktestDataService
from app.services.backtest.backtest_service import BacktestService
from database.connection import SessionLocal
from ..persistence.experiment_persistence_service import ExperimentPersistenceService
from ..models.ga_config import GAConfig

logger = logging.getLogger(__name__)


class AutoStrategyOrchestrationService:
    """
    オートストラテジー統合管理サービス

    戦略テスト、戦略遺伝子復元等の
    統一的な処理を担当します。
    AutoStrategyServiceの機能を直接統合しています。
    """

    def __init__(self, enable_smart_generation: bool = True):
        """
        初期化

        Args:
            enable_smart_generation: SmartConditionGeneratorを使用するか
        """
        # データベースセッションファクトリ
        self.db_session_factory = SessionLocal
        self.enable_smart_generation = enable_smart_generation

        # サービスの初期化
        self.backtest_service: BacktestService
        self.persistence_service: ExperimentPersistenceService

        # 管理マネージャー
        self.experiment_manager: Optional[ExperimentManager] = None

        self._init_services()

    def _init_services(self):
        """
        サービスの初期化

        必要最小限のサービス初期化を行います。
        """
        try:
            # データベースリポジトリの初期化
            with self.db_session_factory() as db:
                from database.repositories.funding_rate_repository import (
                    FundingRateRepository,
                )
                from database.repositories.ohlcv_repository import OHLCVRepository
                from database.repositories.open_interest_repository import (
                    OpenInterestRepository,
                )

                ohlcv_repo = OHLCVRepository(db)
                oi_repo = OpenInterestRepository(db)
                fr_repo = FundingRateRepository(db)

                # バックテストサービスの初期化
                data_service = BacktestDataService(
                    ohlcv_repo=ohlcv_repo, oi_repo=oi_repo, fr_repo=fr_repo
                )
                self.backtest_service = BacktestService(data_service)

            # 永続化サービスの初期化
            self.persistence_service = ExperimentPersistenceService(
                self.db_session_factory, self.backtest_service
            )

            # 実験管理マネージャーの初期化
            self.experiment_manager = ExperimentManager(
                self.backtest_service, self.persistence_service
            )

        except Exception as e:
            logger.error(f"サービス初期化エラー: {e}")
            raise

    async def test_strategy(self, request) -> Dict[str, Any]:
        """
        単一戦略のテスト実行

        指定された戦略遺伝子から戦略を生成し、バックテストを実行します。

        Args:
            request: StrategyTestRequest

        Returns:
            戦略テスト結果
        """
        try:
            # 戦略遺伝子の復元
            from app.services.auto_strategy.models.gene_serialization import (
                GeneSerializer,
            )

            serializer = GeneSerializer()
            strategy_gene = serializer.dict_to_strategy_gene(
                request.strategy_gene, StrategyGene
            )

            # テスト実行
            if not self.experiment_manager:
                return {
                    "success": False,
                    "error": "実験管理マネージャーが初期化されていません。",
                }
            result = self.experiment_manager.test_strategy_generation(
                strategy_gene, request.backtest_config
            )

            if result["success"]:
                return {
                    "success": True,
                    "result": result,
                    "message": "戦略テストが完了しました",
                }
            else:
                return {
                    "success": False,
                    "result": None,
                    "errors": result.get("errors"),
                    "message": "戦略テストに失敗しました",
                }

        except Exception as e:
            logger.error(f"戦略テストエラー: {e}")
            return {
                "success": False,
                "result": None,
                "errors": [str(e)],
                "message": f"戦略テスト中にエラーが発生しました: {str(e)}",
            }

    def validate_experiment_stop(self, experiment_id: str) -> bool:
        """
        実験停止のバリデーション

        Args:
            experiment_id: 実験ID

        Returns:
            停止可能かどうか

        Raises:
            ValueError: 停止できない場合
        """
        if not self.experiment_manager:
            raise ValueError("実験管理マネージャーが初期化されていません。")
        success = self.experiment_manager.stop_experiment(experiment_id)

        if not success:
            logger.warning(
                f"実験 {experiment_id} を停止できませんでした（存在しないか、既に完了している可能性があります）"
            )
            raise ValueError(
                "実験を停止できませんでした（存在しないか、既に完了している可能性があります）"
            )

        return success

    def start_strategy_generation(
        self,
        experiment_id: str,
        experiment_name: str,
        ga_config_dict: Dict[str, Any],
        backtest_config_dict: Dict[str, Any],
        background_tasks: BackgroundTasks,
    ) -> str:
        """
        戦略生成を開始

        Args:
            experiment_id: 実験ID（フロントエンドで生成されたUUID）
            experiment_name: 実験名
            ga_config_dict: GA設定の辞書
            backtest_config_dict: バックテスト設定の辞書
            background_tasks: FastAPIのバックグラウンドタスク

        Returns:
            実験ID（入力されたものと同じ）
        """
        logger.info(f"戦略生成開始: {experiment_name}")

        # 1. GA設定の構築と検証
        try:
            ga_config = GAConfig.from_dict(ga_config_dict)
            is_valid, errors = ga_config.validate()
            if not is_valid:
                raise ValueError(f"無効なGA設定です: {', '.join(errors)}")
        except Exception as e:
            raise ValueError(f"GA設定の構築または検証に失敗しました: {e}")

        # 2. バックテスト設定のシンボル正規化
        backtest_config = backtest_config_dict.copy()
        original_symbol = backtest_config.get("symbol")
        if original_symbol and ":" not in original_symbol:
            normalized_symbol = f"{original_symbol}:USDT"
            backtest_config["symbol"] = normalized_symbol

        # 3. 実験を作成（統合版）
        # フロントエンドから送信されたexperiment_idを使用
        self.persistence_service.create_experiment(
            experiment_id, experiment_name, ga_config, backtest_config
        )

        # 4. GAエンジンを初期化
        if not self.experiment_manager:
            raise RuntimeError("実験管理マネージャーが初期化されていません。")
        self.experiment_manager.initialize_ga_engine(ga_config)

        # 5. 実験をバックグラウンドで開始
        background_tasks.add_task(
            self.experiment_manager.run_experiment,
            experiment_id,
            ga_config,
            backtest_config,
        )

        logger.info(
            f"戦略生成実験のバックグラウンドタスクを追加しました: {experiment_id}"
        )
        return experiment_id

    def get_experiment_progress(self, experiment_id: str) -> Dict[str, Any]:
        """実験の進捗を取得"""
        return self.persistence_service.get_experiment_progress(experiment_id)

    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """実験結果を取得"""
        return self.persistence_service.get_experiment_results(experiment_id)

    def validate_strategy_gene(self, gene: StrategyGene) -> tuple[bool, List[str]]:
        """
        戦略遺伝子の妥当性を検証

        Args:
            gene: 検証する戦略遺伝子

        Returns:
            (is_valid, error_messages)
        """
        if not self.experiment_manager:
            return False, ["実験管理マネージャーが初期化されていません。"]
        return self.experiment_manager.validate_strategy_gene(gene)

    from typing import Optional

    def format_experiment_result(
        self, result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        実験結果のフォーマット

        Args:
            result: 実験結果（None の可能性あり）

        Returns:
            フォーマット済み結果（必ず Dict を返す）
        """
        if result is None:
            return {
                "success": False,
                "message": "実験結果が見つかりませんでした",
                "data": {"result": None},
            }

        # 多目的最適化の結果かどうかを判定
        if "pareto_front" in result and "objectives" in result:
            return {
                "success": True,
                "message": "多目的最適化実験結果を取得しました",
                "data": {
                    "result": result,
                    "pareto_front": result.get("pareto_front"),
                    "objectives": result.get("objectives"),
                },
            }
        else:
            return {
                "success": True,
                "message": "実験結果を取得しました",
                "data": {"result": result},
            }
