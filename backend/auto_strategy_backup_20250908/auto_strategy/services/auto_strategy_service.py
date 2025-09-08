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
from ..config.constants import DEFAULT_SYMBOL

from .experiment_manager import ExperimentManager
from ..config import GAConfig
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
            enable_smart_generation: ConditionGeneratorを使用するか
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
        from app.utils.error_handler import safe_operation

        @safe_operation(context="サービス初期化", is_api_call=False)
        def _init_services_impl():
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

        _init_services_impl()

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
        from app.utils.error_handler import safe_operation

        @safe_operation(context="GA設定構築と検証", is_api_call=True)
        def _validate_ga_config():
            ga_config = GAConfig.from_dict(ga_config_dict)
            is_valid, errors = ga_config.validate()
            if not is_valid:
                raise ValueError(f"無効なGA設定です: {', '.join(errors)}")
            return ga_config

        ga_config = _validate_ga_config()

        # 2. バックテスト設定のシンボル設定
        backtest_config = backtest_config_dict.copy()
        backtest_config["symbol"] = backtest_config.get("symbol", DEFAULT_SYMBOL)

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
        from app.utils.error_handler import safe_operation

        @safe_operation(context="実験一覧取得", is_api_call=False, default_return=[])
        def _list_experiments():
            return self.persistence_service.list_experiments()

        return _list_experiments()


    def stop_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        実験を停止

        Args:
            experiment_id: 実験ID

        Returns:
            停止結果
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(
            context="実験停止",
            is_api_call=False,
            default_return={
                "success": False,
                "message": "実験停止でエラーが発生しました",
            },
        )
        def _stop_experiment():
            if self.experiment_manager:
                # ExperimentManager.stop_experiment()はboolを返すので、Dict形式に変換
                stop_result = self.experiment_manager.stop_experiment(experiment_id)
                if stop_result:
                    return {
                        "success": True,
                        "message": "実験が正常に停止されました",
                    }
                else:
                    return {
                        "success": False,
                        "message": "実験の停止に失敗しました",
                    }
            else:
                return {
                    "success": False,
                    "message": "実験管理マネージャーが初期化されていません",
                }

    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        実験結果を取得

        Args:
            experiment_id: 実験ID

        Returns:
            実験結果の辞書
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(
            context="実験結果取得",
            is_api_call=False,
            default_return={
                "experiment_id": experiment_id,
                "status": "not_found",
                "message": "実験結果が見つかりませんでした",
            },
        )
        def _get_experiment_results():
            if not self.persistence_service:
                return {
                    "experiment_id": experiment_id,
                    "status": "error",
                    "message": "永続化サービスが初期化されていません",
                }

            # 永続化サービスから結果を取得
            results = self.persistence_service.get_experiment_results(experiment_id)

            if results:
                return {
                    "experiment_id": experiment_id,
                    "status": "completed",
                    **results,
                }

            # 実験が存在するか確認
            experiment_info = self.persistence_service.get_experiment_status(experiment_id)
            if experiment_info and experiment_info.get("status") == "running":
                return {
                    "experiment_id": experiment_id,
                    "status": "running",
                    "message": "実験は実行中です",
                    "progress": experiment_info.get("progress", 0),
                }

            return {
                "experiment_id": experiment_id,
                "status": "not_found",
                "message": "実験結果が見つかりませんでした",
            }

        return _get_experiment_results()

    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """
        実験ステータスを取得

        Args:
            experiment_id: 実験ID

        Returns:
            実験ステータスの辞書
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(
            context="実験ステータス取得",
            is_api_call=False,
            default_return={
                "experiment_id": experiment_id,
                "status": "error",
                "message": "ステータス取得中にエラーが発生しました",
            },
        )
        def _get_experiment_status():
            if not self.persistence_service:
                return {
                    "experiment_id": experiment_id,
                    "status": "error",
                    "message": "永続化サービスが初期化されていません",
                }

            # 永続化サービスからステータスを取得
            status_info = self.persistence_service.get_experiment_status(experiment_id)

            if status_info:
                return {
                    "experiment_id": experiment_id,
                    **status_info,
                }

            return {
                "experiment_id": experiment_id,
                "status": "not_found",
                "message": "実験が見つかりませんでした",
            }

        return _get_experiment_status()
        return _stop_experiment()
