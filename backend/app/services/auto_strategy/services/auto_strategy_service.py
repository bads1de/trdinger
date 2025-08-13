"""
自動戦略生成サービス

GA実行、進捗管理、結果保存、戦略テストを統合的に管理します。
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks

from app.services.backtest.backtest_data_service import BacktestDataService
from app.services.backtest.backtest_service import BacktestService
from database.connection import SessionLocal

from .experiment_manager import ExperimentManager
from ..models.ga_config import GAConfig
from .experiment_persistence_service import ExperimentPersistenceService

logger = logging.getLogger(__name__)


class AutoStrategyService:
    """
    自動戦略生成サービス

    GA実行、進捗管理、結果保存、戦略テストを統合的に管理します。
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

    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        実験一覧を取得

        Returns:
            実験一覧のリスト
        """
        try:
            return self.persistence_service.list_experiments()
        except Exception as e:
            logger.error(f"実験一覧取得エラー: {e}")
            return []

    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """
        実験ステータスを取得

        Args:
            experiment_id: 実験ID

        Returns:
            実験ステータス情報
        """
        try:
            # get_experiment_infoメソッドを使用
            experiment_info = self.persistence_service.get_experiment_info(
                experiment_id
            )
            if experiment_info:
                return {
                    "status": experiment_info.get("status", "unknown"),
                    "progress": experiment_info.get("progress", 0),
                    "message": experiment_info.get("message", "実験情報を取得しました"),
                    "experiment_info": experiment_info,
                }
            else:
                return {"status": "not_found", "message": "実験が見つかりません"}
        except Exception as e:
            logger.error(f"実験ステータス取得エラー: {e}")
            return {"status": "error", "message": str(e)}

    def stop_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        実験を停止

        Args:
            experiment_id: 実験ID

        Returns:
            停止結果
        """
        try:
            if self.experiment_manager:
                return self.experiment_manager.stop_experiment(experiment_id)
            else:
                return {
                    "success": False,
                    "message": "実験管理マネージャーが初期化されていません",
                }
        except Exception as e:
            logger.error(f"実験停止エラー: {e}")
            return {"success": False, "message": str(e)}

    def get_default_config(self) -> Dict[str, Any]:
        """
        デフォルト設定を取得

        Returns:
            デフォルト設定
        """
        try:
            return GAConfig().to_dict()
        except Exception as e:
            logger.error(f"デフォルト設定取得エラー: {e}")
            return {}

    def get_presets(self) -> Dict[str, Any]:
        """
        プリセット設定を取得

        Returns:
            プリセット設定
        """
        try:
            return GAConfig.get_presets()
        except Exception as e:
            logger.error(f"プリセット取得エラー: {e}")
            return {}

    async def test_strategy(self, request) -> Dict[str, Any]:
        """
        単一戦略のテスト実行

        指定された戦略遺伝子から戦略を生成し、バックテストを実行します。

        Args:
            request: StrategyTestRequest

        Returns:
            テスト結果
        """
        try:
            logger.info("戦略テスト開始")

            # 戦略遺伝子の復元
            from ..models.gene_serialization import GeneSerializer

            strategy_gene = GeneSerializer.from_dict(request.strategy_gene)

            # バックテスト設定の正規化
            backtest_config = request.backtest_config.copy()
            original_symbol = backtest_config.get("symbol")
            if original_symbol and ":" not in original_symbol:
                normalized_symbol = f"{original_symbol}:USDT"
                backtest_config["symbol"] = normalized_symbol

            # バックテスト実行
            result = self.backtest_service.run_backtest_with_gene(
                strategy_gene, backtest_config
            )

            logger.info("戦略テスト完了")
            return {
                "success": True,
                "result": result,
                "message": "戦略テストが正常に完了しました",
            }

        except Exception as e:
            logger.error(f"戦略テストエラー: {e}", exc_info=True)
            return {
                "success": False,
                "errors": [str(e)],
                "message": f"戦略テストに失敗しました: {str(e)}",
            }
