"""
バックテスト実行エンジン

"""

import logging
import warnings
from datetime import datetime
from typing import Any, Dict, Type

import pandas as pd
from backtesting import Backtest, Strategy

from ..backtest_data_service import BacktestDataService

logger = logging.getLogger(__name__)


class BacktestExecutionError(Exception):
    """バックテスト実行エラー"""


class BacktestExecutor:
    """
    バックテスト実行エンジン

    backtesting.pyライブラリを使用したバックテスト実行を専門に担当します。
    """

    def __init__(self, data_service: BacktestDataService):
        """
        初期化

        Args:
            data_service: データサービス
        """
        self.data_service = data_service

    def execute_backtest(
        self,
        strategy_class: Type[Strategy],
        strategy_parameters: Dict[str, Any],
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float,
        commission_rate: float,
    ) -> Any:
        """
        バックテストを実行

        Args:
            strategy_class: 戦略クラス
            strategy_parameters: 戦略パラメータ
            symbol: 取引ペア
            timeframe: 時間軸
            start_date: 開始日時
            end_date: 終了日時
            initial_capital: 初期資金
            commission_rate: 手数料率

        Returns:
            バックテスト統計結果

        Raises:
            BacktestExecutionError: バックテスト実行に失敗した場合
        """
        try:
            # データ取得
            data = self._get_backtest_data(symbol, timeframe, start_date, end_date)

            # バックテスト設定
            bt = self._create_backtest_instance(
                data, strategy_class, initial_capital, commission_rate
            )

            # バックテスト実行
            stats = self._run_backtest(bt, strategy_parameters)

            return stats

        except Exception as e:
            logger.error(f"バックテスト実行エラー: {e}")
            raise BacktestExecutionError(f"バックテストの実行に失敗しました: {e}")

    def _get_backtest_data(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """バックテスト用データを取得"""
        try:
            data = self.data_service.get_data_for_backtest(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )

            if data.empty:
                raise BacktestExecutionError(
                    f"{symbol} {timeframe}のデータが見つかりませんでした。"
                )

            return data

        except Exception as e:
            raise BacktestExecutionError(f"データ取得に失敗しました: {e}")

    def _create_backtest_instance(
        self,
        data: pd.DataFrame,
        strategy_class: Type[Strategy],
        initial_capital: float,
        commission_rate: float,
    ) -> Backtest:
        """バックテストインスタンスを作成"""
        try:
            bt = Backtest(
                data,
                strategy_class,
                cash=initial_capital,
                commission=commission_rate,
                exclusive_orders=False,  # 複数ポジション許可（制約緩和）
                trade_on_close=False,  # 現在価格で取引（制約緩和）
                hedging=True,  # ヘッジング有効化（制約緩和）
                margin=0.01,  # マージン要件を大幅に緩和（1%）
            )

            return bt

        except Exception as e:
            raise BacktestExecutionError(
                f"バックテストインスタンスの作成に失敗しました: {e}"
            )

    def _run_backtest(self, bt: Backtest, strategy_parameters: Dict[str, Any]) -> Any:
        """バックテストを実行"""
        try:

            # 警告を一時的に無効化
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                stats = bt.run(**strategy_parameters)

            return stats

        except Exception as e:
            raise BacktestExecutionError(
                f"バックテスト実行中にエラーが発生しました: {e}"
            )

    def get_supported_strategies(self) -> Dict[str, Any]:
        """
        サポートされている戦略一覧を取得

        Returns:
            戦略一覧
        """
        # 現在はオートストラテジーのみサポート
        return {
            "auto_strategy": {
                "name": "オートストラテジー",
                "description": "遺伝的アルゴリズムで生成された戦略",
                "parameters": {
                    "strategy_gene": {
                        "type": "dict",
                        "required": True,
                        "description": "戦略遺伝子",
                    }
                },
            }
        }
