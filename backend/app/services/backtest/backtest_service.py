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
from .conversion.backtest_result_converter import (
    BacktestResultConversionError,
    BacktestResultConverter,
)
from .execution.backtest_executor import BacktestExecutionError, BacktestExecutor
from .factories.strategy_class_factory import (
    StrategyClassCreationError,
    StrategyClassFactory,
)
from .validation.backtest_config_validator import (
    BacktestConfigValidationError,
    BacktestConfigValidator,
)

logger = logging.getLogger(__name__)


class BacktestService:
    """
    バックテスト実行サービス
    """

    def __init__(self, data_service: Optional[BacktestDataService] = None):
        """
        初期化

        Args:
            data_service: データ変換サービス（テスト時にモックを注入可能）
        """
        self.data_service = data_service
        self._db_session = None  # DBセッション保持用
        self._validator = BacktestConfigValidator()
        self._strategy_factory = StrategyClassFactory()
        self._result_converter = BacktestResultConverter()
        self._executor = None  # 遅延初期化

    def run_backtest(
        self, config: Dict[str, Any], preloaded_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        バックテストを実行

        リファクタリング後の実装では、各専門サービスに処理を委譲します。

        Args:
            config: バックテスト設定
                - strategy_name: 戦略名
                - symbol: 取引ペア
                - timeframe: 時間軸
                - start_date: 開始日時
                - end_date: 終了日時
                - initial_capital: 初期資金
                - commission_rate: 手数料率
                - strategy_config: 戦略固有の設定

        Returns:
            バックテスト結果の辞書

        Raises:
            BacktestConfigValidationError: 設定が無効な場合
            StrategyClassCreationError: 戦略クラス生成に失敗した場合
            BacktestExecutionError: バックテスト実行に失敗した場合
            BacktestResultConversionError: 結果変換に失敗した場合
        """
        try:
            # 1. 設定の検証
            self._validator.validate_config(config)

            # 2. データサービスの初期化
            self.ensure_data_service_initialized()

            # 3. 実行エンジンの初期化
            self._ensure_executor_initialized()

            # 4. 日付の正規化
            start_date = self._normalize_date(config["start_date"])
            end_date = self._normalize_date(config["end_date"])

            # 5. 戦略クラス取得または生成
            if "strategy_class" in config:
                # GAエンジンから直接戦略クラスが渡された場合
                strategy_class = config["strategy_class"]
                strategy_parameters = {}
            else:
                # 通常のstrategy_configから戦略クラスを生成する場合
                strategy_class = self._strategy_factory.create_strategy_class(
                    config["strategy_config"]
                )
                strategy_parameters = self._strategy_factory.get_strategy_parameters(
                    config["strategy_config"]
                )

            # 6. バックテスト実行
            if self._executor is None:
                raise BacktestExecutionError("実行エンジンが初期化されていません")
            stats = self._executor.execute_backtest(
                strategy_class=strategy_class,
                strategy_parameters=strategy_parameters,
                symbol=config["symbol"],
                timeframe=config["timeframe"],
                start_date=start_date,
                end_date=end_date,
                initial_capital=config["initial_capital"],
                commission_rate=config["commission_rate"],
                preloaded_data=preloaded_data,
            )

            # 7. 結果をデータベース形式に変換
            config_json = {
                "strategy_config": config.get("strategy_config", {}),
                "commission_rate": config.get("commission_rate", 0.001),
            }

            result = self._result_converter.convert_backtest_results(
                stats=stats,
                strategy_name=config["strategy_name"],
                symbol=config["symbol"],
                timeframe=config["timeframe"],
                initial_capital=config["initial_capital"],
                start_date=config["start_date"],
                end_date=config["end_date"],
                config_json=config_json,
            )

            return result

        except (
            BacktestConfigValidationError,
            StrategyClassCreationError,
            BacktestExecutionError,
            BacktestResultConversionError,
        ) as e:
            # 専用例外はそのまま再発生
            logger.error(f"バックテスト実行エラー: {e}")
            raise
        except Exception as e:
            logger.error(f"予期しないエラーが発生しました: {e}", exc_info=True)
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

    def _ensure_executor_initialized(self) -> None:
        """実行エンジンの初期化を確保"""
        if self._executor is None:
            if self.data_service is None:
                raise BacktestExecutionError(
                    "データサービスが初期化されていません。バックテストを実行できません。"
                )

            try:
                self._executor = BacktestExecutor(self.data_service)
                logger.info("バックテスト実行エンジンを初期化しました")
            except Exception as e:
                logger.error(f"バックテスト実行エンジンの初期化に失敗しました: {e}")
                raise BacktestExecutionError(f"実行エンジンの初期化に失敗しました: {e}")

    def _normalize_date(self, date_value: Any) -> datetime:
        """日付値をdatetimeオブジェクトに正規化"""
        if isinstance(date_value, datetime):
            return date_value
        elif isinstance(date_value, str):
            return datetime.fromisoformat(date_value.replace("Z", "+00:00"))
        else:
            raise ValueError(f"サポートされていない日付形式: {type(date_value)}")

    def get_supported_strategies(self) -> Dict[str, Any]:
        """
        サポートされている戦略一覧を取得

        Returns:
            戦略一覧
        """
        self._ensure_executor_initialized()
        if self._executor is None:
            raise BacktestExecutionError("実行エンジンが初期化されていません")
        return self._executor.get_supported_strategies()

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
            request: BacktestRequestオブジェクトまたは辞書
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
