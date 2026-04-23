"""
バックテストオーケストレーター

バックテストの実行フローを制御します。
DB操作や永続化の責務を持たず、純粋なバックテストの実行と結果生成に集中します。
"""

import copy
import logging
from typing import Any, Dict, Optional

import pandas as pd
from pydantic import ValidationError

from ..config.backtest_config import (
    BacktestRunConfig,
    BacktestRunConfigValidationError,
)
from ..conversion.backtest_result_converter import (
    BacktestResultConversionError,
    BacktestResultConverter,
)
from ..factories.strategy_class_factory import (
    StrategyClassCreationError,
    StrategyClassFactory,
)
from ..services.backtest_data_service import BacktestDataService
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
        self._strategy_factory = StrategyClassFactory()
        self._result_converter = BacktestResultConverter()
        self._executor = BacktestExecutor(data_service)

    def run(
        self,
        config: Dict[str, Any],
        preloaded_data: Optional[pd.DataFrame] = None,
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
            working_config = copy.deepcopy(config)
            spread_value = working_config.get("spread")
            slippage_value = working_config.get("slippage")
            if spread_value is None and slippage_value is not None:
                working_config["spread"] = slippage_value
            elif slippage_value is None and spread_value is not None:
                working_config["slippage"] = spread_value

            # 高速化フラグの確認
            skip_validation = working_config.pop("_skip_validation", False)
            include_raw_stats = working_config.pop("_include_raw_stats", False)

            if skip_validation:
                # バリデーションスキップ（GAなど信頼できる内部呼び出し用）
                # model_constructでバリデーションをスキップしつつPydanticモデルインスタンスを取得
                backtest_config = BacktestRunConfig.model_construct(
                    **working_config
                )
                # datetime 変換を保証する（文字列の場合は変換）
                if isinstance(backtest_config.start_date, str):
                    backtest_config.start_date = pd.to_datetime(
                        backtest_config.start_date
                    ).to_pydatetime()
                if isinstance(backtest_config.end_date, str):
                    backtest_config.end_date = pd.to_datetime(
                        backtest_config.end_date
                    ).to_pydatetime()
                strategy_config_dict = working_config["strategy_config"]
            else:
                # 1. 設定の検証とモデル変換 (Pydantic)
                try:
                    backtest_config = BacktestRunConfig(**working_config)
                except ValidationError as e:
                    raise BacktestRunConfigValidationError(
                        f"設定が無効です: {e}"
                    )

                # 2. 戦略クラス取得または生成
                # StrategyClassFactoryはまだ辞書を期待しているため、一部辞書に戻す
                # strategy_configオブジェクトを辞書に変換
                strategy_config_dict = (
                    backtest_config.strategy_config.model_dump()
                )

            # strategy_class が config に直接含まれている場合の対応（GAエンジンからの直接渡しなど）
            # Pydanticモデルには含まれていないため、元のconfig辞書を確認
            if "strategy_class" in working_config:
                strategy_class = working_config["strategy_class"]
                strategy_parameters = {}
            else:
                strategy_class = self._strategy_factory.create_strategy_class(
                    strategy_config_dict
                )
                strategy_parameters = (
                    self._strategy_factory.get_strategy_parameters(
                        strategy_config_dict
                    )
                )

            # 3. バックテスト実行
            stats = self._executor.execute_backtest(
                strategy_class=strategy_class,
                strategy_parameters=strategy_parameters,
                symbol=backtest_config.symbol,
                timeframe=backtest_config.timeframe,
                start_date=backtest_config.start_date,
                end_date=backtest_config.end_date,
                initial_capital=backtest_config.initial_capital,
                commission_rate=backtest_config.commission_rate,
                slippage=backtest_config.spread,
                leverage=backtest_config.leverage,
                preloaded_data=preloaded_data,
            )

            # 4. 結果をデータベース形式に変換
            config_json = {
                "strategy_config": strategy_config_dict,
                "commission_rate": backtest_config.commission_rate,
                "spread": backtest_config.spread,
                "slippage": backtest_config.slippage,
                "leverage": backtest_config.leverage,
            }

            result = self._result_converter.convert_backtest_results(
                stats=stats,
                strategy_name=backtest_config.strategy_name,
                symbol=backtest_config.symbol,
                timeframe=backtest_config.timeframe,
                initial_capital=backtest_config.initial_capital,
                start_date=backtest_config.start_date,
                end_date=backtest_config.end_date,
                config_json=config_json,
            )

            if include_raw_stats:
                result["_raw_stats"] = stats

            return result

        except (
            BacktestRunConfigValidationError,
            StrategyClassCreationError,
            BacktestExecutionError,
            BacktestResultConversionError,
        ) as e:
            logger.error(f"バックテスト実行エラー: {e}")
            raise
        except Exception as e:
            logger.error(f"予期しないエラーが発生しました: {e}", exc_info=True)
            raise

    def get_supported_strategies(self) -> Dict[str, Any]:
        """サポートされている戦略一覧を取得"""
        return self._executor.get_supported_strategies()
