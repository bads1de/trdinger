"""
バックテスト実行エンジン

"""

import logging
import warnings
from datetime import datetime
from typing import Any, Dict, Optional, Type

import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import FractionalBacktest

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
        slippage: float = 0.0,
        leverage: float = 1.0,
        preloaded_data: Optional[pd.DataFrame] = None,
    ) -> Any:
        """
        指定された戦略とパラメータでバックテスト・シミュレーションを実行

        `backtesting.py` の `FractionalBacktest` をベースに、仮想通貨特有の
        レバレッジ設定やマージン要件を緩和した状態で実行し、トレード統計
        （勝率、ドローダウン、総利益等）を算出します。

        Args:
            strategy_class: 実行する `backtesting.Strategy` 継承クラス
            strategy_parameters: 指標期間等の戦略設定パラメータ
            symbol: 取引ペア
            timeframe: 時間軸
            start_date: 検証開始日時
            end_date: 検証終了日時
            initial_capital: 初期証拠金
            commission_rate: 手数料率（例: 0.0006 for 0.06%）
            slippage: スリッページ率（例: 0.0001 for 0.01%）。簡易的に手数料に加算されます。
            leverage: レバレッジ倍率（例: 1.0でレバレッジなし）。マージン率に変換されます。
            preloaded_data: 外部から提供されたOHLCVデータがある場合に使用

        Returns:
            パフォーマンス統計を含む `backtesting.stats` オブジェクト

        Raises:
            BacktestExecutionError: シミュレーションの初期化または実行中に異常が発生した場合
        """
        try:
            # データ取得
            if preloaded_data is not None:
                data = preloaded_data
            else:
                data = self._get_backtest_data(symbol, timeframe, start_date, end_date)

            # バックテスト設定
            bt = self._create_backtest_instance(
                data,
                strategy_class,
                initial_capital,
                commission_rate,
                slippage,
                leverage,
                symbol,
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
                logger.error(
                    f"BacktestExecutor - No data found for {symbol} {timeframe}"
                )
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
        slippage: float,
        leverage: float,
        symbol: str,
    ) -> Backtest:
        """バックテストインスタンスを作成"""
        try:
            # backtesting.pyライブラリが大文字のカラム名を期待するため変換
            data = data.copy()
            data.columns = data.columns.str.capitalize()

            # スリッページを簡易的に手数料に加算（backtesting.pyの標準機能でサポートが薄いため）
            effective_commission = commission_rate + slippage

            # レバレッジからマージン率を計算（例: レバレッジ10倍 -> マージン0.1）
            # 0除算防止のためmax(1.0, ...)を使用
            margin = 1.0 / max(1.0, leverage)

            bt = FractionalBacktest(
                data,
                strategy_class,
                cash=initial_capital,
                commission=effective_commission,
                exclusive_orders=False,  # 複数ポジション許可（制約緩和）
                trade_on_close=False,  # 現在価格で取引（制約緩和）
                hedging=True,  # ヘッジング有効化（制約緩和）
                margin=margin,
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
