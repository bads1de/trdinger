"""
バックテスト実行サービス

"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from sqlalchemy.orm import Session

from database.connection import get_db
from database.repositories.backtest_result_repository import BacktestResultRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository

from .backtest_data_service import BacktestDataService
from .execution.backtest_executor import BacktestExecutionError
from .execution.backtest_orchestrator import BacktestOrchestrator

logger = logging.getLogger(__name__)


class BacktestService:
    """
    バックテスト実行サービス

    責務:
    1. DBセッションとデータサービスの初期化・管理
    2. バックテストオーケストレーターへの処理委譲
    3. 結果のデータベースへの保存
    """

    def __init__(self, data_service: Optional[BacktestDataService] = None):
        """
        初期化

        Args:
            data_service: データ変換サービス（テスト時にモックを注入可能）
        """
        self.data_service = data_service
        self._db_session = None  # DBセッション保持用
        self._orchestrator = None  # 遅延初期化

    def run_backtest(
        self, config: Dict[str, Any], preloaded_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        バックテストを実行

        バックテストオーケストレーターに処理を委譲します。

        Args:
            config: バックテスト設定
            preloaded_data: 事前にロードされたデータ（オプション）

        Returns:
            バックテスト結果の辞書
        """
        try:
            # 1. データサービスの初期化
            self.ensure_data_service_initialized()

            # 2. オーケストレーターの初期化
            self._ensure_orchestrator_initialized()

            # 3. オーケストレーターに委譲
            if self._orchestrator is None:
                raise BacktestExecutionError("オーケストレーターが初期化されていません")

            return self._orchestrator.run(config, preloaded_data)

        except Exception as e:
            logger.error(f"バックテスト実行エラー: {e}")
            raise

    def ensure_data_service_initialized(self) -> None:
        """データサービスの初期化を確保"""
        if self.data_service is None:
            # 新しいDBセッションを作成し、保持する
            self._db_session = next(get_db())
            try:
                ohlcv_repo = OHLCVRepository(self._db_session)
                oi_repo = OpenInterestRepository(self._db_session)
                fr_repo = FundingRateRepository(self._db_session)
                self.data_service = BacktestDataService(
                    ohlcv_repo=ohlcv_repo, oi_repo=oi_repo, fr_repo=fr_repo
                )
                logger.info("バックテストデータサービスを初期化しました")
            except Exception as e:
                logger.error(f"バックテストデータサービスの初期化に失敗しました: {e}")
                raise BacktestExecutionError(
                    f"データサービスの初期化に失敗しました: {e}"
                )

    def _ensure_orchestrator_initialized(self) -> None:
        """オーケストレーターの初期化を確保"""
        if self._orchestrator is None:
            if self.data_service is None:
                raise BacktestExecutionError(
                    "データサービスが初期化されていません。バックテストを実行できません。"
                )

            try:
                self._orchestrator = BacktestOrchestrator(self.data_service)
                logger.info("バックテストオーケストレーターを初期化しました")
            except Exception as e:
                logger.error(
                    f"バックテストオーケストレーターの初期化に失敗しました: {e}"
                )
                raise BacktestExecutionError(
                    f"オーケストレーターの初期化に失敗しました: {e}"
                )

    def get_supported_strategies(self) -> Dict[str, Any]:
        """
        サポートされている戦略一覧を取得

        Returns:
            戦略一覧
        """
        self.ensure_data_service_initialized()
        self._ensure_orchestrator_initialized()
        if self._orchestrator is None:
            raise BacktestExecutionError("オーケストレーターが初期化されていません")
        return self._orchestrator.get_supported_strategies()

    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        if self._db_session is not None:
            self._db_session.close()
            self._db_session = None
            logger.info("DBセッションをクリーンアップしました")

    def execute_and_save_backtest(self, request, db_session: Session) -> Dict[str, Any]:
        """
        バックテストを実行し、結果をデータベースに保存

        Args:
            request: BacktestConfigオブジェクトまたは辞書
            db_session: データベースセッション

        Returns:
            実行結果の辞書
        """
        try:
            # リクエストから設定を作成（辞書とPydanticモデルの両方に対応）
            if isinstance(request, dict):
                # 辞書の場合
                config = {
                    "strategy_name": request["strategy_name"],
                    "symbol": request["symbol"],
                    "timeframe": request["timeframe"],
                    "start_date": request["start_date"],
                    "end_date": request["end_date"],
                    "initial_capital": request["initial_capital"],
                    "commission_rate": request["commission_rate"],
                    "strategy_config": request["strategy_config"],
                }
            else:
                # Pydanticモデルの場合
                config = {
                    "strategy_name": request.strategy_name,
                    "symbol": request.symbol,
                    "timeframe": request.timeframe,
                    "start_date": request.start_date,
                    "end_date": request.end_date,
                    "initial_capital": request.initial_capital,
                    "commission_rate": request.commission_rate,
                    "strategy_config": request.strategy_config.model_dump(),
                }

            # バックテストを実行
            result = self.run_backtest(config)

            # 結果をデータベースに保存
            backtest_repo = BacktestResultRepository(db_session)
            saved_result = backtest_repo.save_backtest_result(result)

            return {"success": True, "result": saved_result}

        except Exception as e:
            logger.error(f"バックテスト実行・保存エラー: {e}", exc_info=True)
            return {"success": False, "error": str(e), "status_code": 500}
