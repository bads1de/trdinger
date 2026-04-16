"""
自動戦略生成サービス

GA実行、進捗管理、結果保存、戦略テストを統合的に管理します。
"""

import logging
from typing import Any, Dict, List, Optional

from app.services.backtest.services.backtest_service import BacktestService
from app.utils.error_handler import ErrorHandler
from database.connection import SessionLocal

from ..config import GAConfig
from ..config.constants import DEFAULT_SYMBOL
from .experiment_application_service import (
    ExperimentApplicationService,
    TaskScheduler,
)
from .experiment_manager import ExperimentManager
from .experiment_persistence_service import ExperimentPersistenceService

logger = logging.getLogger(__name__)


class AutoStrategyService:
    """
    自動戦略生成サービス

    GA実行、進捗管理、結果保存、戦略テストを統合的に管理します。
    """

    def __init__(
        self,
        backtest_service: Optional[BacktestService] = None,
        persistence_service: Optional[ExperimentPersistenceService] = None,
        experiment_manager: Optional[ExperimentManager] = None,
        experiment_application_service: Optional[ExperimentApplicationService] = None,
    ):
        """
        初期化

        Args:
            backtest_service: バックテストサービス（DI対応、未指定時は内部生成）
            persistence_service: 永続化サービス（DI対応、未指定時は内部生成）
            experiment_manager: 実験管理マネージャー（DI対応、未指定時は内部生成）
            experiment_application_service: 実験アプリケーションサービス（DI対応、未指定時は内部生成）
        """
        # データベースセッションファクトリ
        self.db_session_factory = SessionLocal

        if (
            backtest_service
            and persistence_service
            and experiment_manager
            and experiment_application_service
        ):
            # DI モード: 全依存が注入された場合
            self.backtest_service = backtest_service
            self.persistence_service = persistence_service
            self.experiment_manager = experiment_manager
            self.experiment_application_service = experiment_application_service
        else:
            # 従来モード: 内部初期化
            self._init_services()

    def _init_services(self) -> None:
        """
        必要な内部サービスを遅延初期化します。
        """
        try:
            # 引数なしで初期化することで、BacktestServiceが実行時に自らセッションを管理するようにする
            self.backtest_service = BacktestService()
            self.persistence_service = ExperimentPersistenceService(
                self.db_session_factory
            )
            self.experiment_manager = ExperimentManager(
                self.backtest_service, self.persistence_service
            )
            self.experiment_application_service = ExperimentApplicationService(
                self.experiment_manager,
                self.persistence_service,
            )
        except Exception as e:
            logger.error("エラー in サービス初期化: %s", e)
            raise

    def start_strategy_generation(
        self,
        experiment_id: str,
        experiment_name: str,
        ga_config_dict: Dict[str, Any],
        backtest_config_dict: Dict[str, Any],
        task_scheduler: TaskScheduler,
    ) -> str:
        """
        戦略生成を開始

        Args:
            experiment_id: 実験ID（フロントエンドで生成されたUUID）
            experiment_name: 実験名
            ga_config_dict: GA設定の辞書
            backtest_config_dict: バックテスト設定の辞書
            task_scheduler: 実行タスクを登録する scheduler

        Returns:
            実験ID（入力されたものと同じ）
        """
        logger.info(f"戦略生成開始: {experiment_name}")

        # 1. GA設定の構築と検証
        ga_config = self._prepare_ga_config(ga_config_dict)

        # 2. バックテスト設定の準備
        backtest_config = self._prepare_backtest_config(backtest_config_dict)

        # 3. 実験の作成
        if not self.experiment_application_service:
            raise RuntimeError("実験 application service が初期化されていません。")

        self.experiment_application_service.start_experiment(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            ga_config=ga_config,
            backtest_config=backtest_config,
            task_scheduler=task_scheduler,
        )

        logger.info(
            f"戦略生成実験のバックグラウンドタスクを追加しました: {experiment_id}"
        )

        return experiment_id

    def _prepare_ga_config(self, settings: Dict[str, Any]) -> GAConfig:
        """
        GAの設定オブジェクトを準備します。

        入力された設定値をバリデートし、デフォルト値とマージして
        GAConfigインスタンスを生成します。

        Args:
            settings: ユーザー定義の設定

        Returns:
            初期化されたGAConfig
        """
        try:
            ga_config = GAConfig.from_dict(settings)
            from ..config import ConfigValidator

            is_valid, errors = ConfigValidator.validate(ga_config)
            if not is_valid:
                raise ValueError(f"無効なGA設定です: {', '.join(errors)}")
            return ga_config
        except Exception as e:
            raise ErrorHandler.handle_api_error(e, context="GA設定構築と検証")

    def _prepare_backtest_config(
        self, backtest_config_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        バックテスト設定の最終調整

        Args:
            backtest_config_dict: ユーザー入力の設定辞書

        Returns:
            シンボル等のデフォルト値が補完された設定辞書
        """
        backtest_config = backtest_config_dict.copy()
        backtest_config["symbol"] = backtest_config.get("symbol", DEFAULT_SYMBOL)
        return backtest_config

    def _create_experiment(
        self,
        experiment_id: str,
        experiment_name: str,
        ga_config: GAConfig,
        backtest_config: Dict[str, Any],
    ):
        """
        データベースに実験レコードを作成

        Args:
            experiment_id: 実験の一意識別子
            experiment_name: 実験の表示名
            ga_config: GA設定
            backtest_config: バックテスト設定
        """
        # フロントエンドから送信されたexperiment_idを使用
        self._get_experiment_application_service().create_experiment(
            experiment_id, experiment_name, ga_config, backtest_config
        )

    def _initialize_ga_engine(self, experiment_id: str, ga_config: GAConfig):
        """
        GAエンジンを初期化

        Args:
            experiment_id: 実験ID
            ga_config: 実験で使用するGA設定
        """
        self._get_experiment_application_service().initialize_ga_engine(
            experiment_id, ga_config
        )

    def _start_experiment_in_background(
        self,
        experiment_id: str,
        ga_config: GAConfig,
        backtest_config: Dict[str, Any],
        task_scheduler: TaskScheduler,
    ):
        """実験をバックグラウンドタスクで開始する"""
        self._get_experiment_application_service().schedule_experiment(
            experiment_id,
            ga_config,
            backtest_config,
            task_scheduler,
        )

    def _get_experiment_application_service(self) -> ExperimentApplicationService:
        """初期化済みの ExperimentApplicationService を返す。"""
        if not self.experiment_application_service:
            raise RuntimeError("実験 application service が初期化されていません。")
        return self.experiment_application_service

    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        実験一覧を取得

        Returns:
            実験一覧のリスト
        """
        try:
            if not self.experiment_application_service:
                return []
            return self.experiment_application_service.list_experiments()
        except Exception as e:
            logger.error("エラー in 実験一覧取得: %s", e)
            return []

    def get_experiment_detail(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        実験詳細を取得（進捗情報を含む）

        Args:
            experiment_id: 実験ID

        Returns:
            実験詳細情報、存在しない場合はNone
        """
        try:
            if not self.experiment_application_service:
                return None
            return self.experiment_application_service.get_experiment_detail(
                experiment_id
            )
        except Exception as e:
            logger.error("エラー in 実験詳細取得: %s", e)
            return None

    def stop_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        実験を停止

        Args:
            experiment_id: 実験ID

        Returns:
            停止結果
        """
        try:
            if not self.experiment_application_service:
                return {
                    "success": False,
                    "message": "実験 application service が初期化されていません",
                }
            return self.experiment_application_service.stop_experiment(experiment_id)
        except Exception as e:
            logger.error("エラー in 実験停止: %s", e)
            return {
                "success": False,
                "message": "実験停止でエラーが発生しました",
            }
