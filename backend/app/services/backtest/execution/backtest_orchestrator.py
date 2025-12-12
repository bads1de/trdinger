"""
バックテストオーケストレーター

バックテストの実行フローを制御します。
DB操作や永続化の責務を持たず、純粋なバックテストの実行と結果生成に集中します。
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from ..backtest_data_service import BacktestDataService
from ..conversion.backtest_result_converter import (
    BacktestResultConversionError,
    BacktestResultConverter,
)
from ..factories.strategy_class_factory import (
    StrategyClassCreationError,
    StrategyClassFactory,
)
from ..validation.backtest_config_validator import (
    BacktestConfigValidationError,
    BacktestConfigValidator,
)
from .backtest_executor import BacktestExecutionError, BacktestExecutor

logger = logging.getLogger(__name__)


class BacktestOrchestrator:
    """
    バックテストオーケストレーター
    """

    def __init__(self, data_service: BacktestDataService):
        """
        初期化

        Args:
            data_service: データサービス（必須）
        """
        if data_service is None:
            raise ValueError("BacktestDataService is required")

        self.data_service = data_service
        self._validator = BacktestConfigValidator()
        self._strategy_factory = StrategyClassFactory()
        self._result_converter = BacktestResultConverter()
        self._executor = BacktestExecutor(data_service)

    def run(
        self, config: Dict[str, Any], preloaded_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        バックテストを実行

        Args:
            config: バックテスト設定
            preloaded_data: 事前にロードされたデータ（オプション）

        Returns:
            バックテスト結果の辞書
        """
        try:
            # 1. 設定の検証
            self._validator.validate_config(config)

            # 2. 日付の正規化
            start_date = self._normalize_date(config["start_date"])
            end_date = self._normalize_date(config["end_date"])

            # 3. 戦略クラス取得または生成
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

            # 4. バックテスト実行
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

            # 5. 結果をデータベース形式に変換
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
            logger.error(f"バックテスト実行エラー: {e}")
            raise
        except Exception as e:
            logger.error(f"予期しないエラーが発生しました: {e}", exc_info=True)
            raise

    def _normalize_date(self, date_value: Any) -> datetime:
        """日付値をdatetimeオブジェクトに正規化"""
        if isinstance(date_value, datetime):
            return date_value
        elif isinstance(date_value, str):
            return datetime.fromisoformat(date_value.replace("Z", "+00:00"))
        else:
            raise ValueError(f"サポートされていない日付形式: {type(date_value)}")

    def get_supported_strategies(self) -> Dict[str, Any]:
        """サポートされている戦略一覧を取得"""
        return self._executor.get_supported_strategies()
