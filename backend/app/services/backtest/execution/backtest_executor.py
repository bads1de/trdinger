"""
バックテスト実行エンジン

"""

import logging
import warnings
from copy import deepcopy
from datetime import datetime
from typing import Dict, Optional, Type

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import FractionalBacktest

from app.services.auto_strategy.strategies.universal_strategy import (
    StrategyEarlyTermination,
)

from ..config.constants import SUPPORTED_STRATEGIES
from ..services.backtest_data_service import BacktestDataService
from ..shared import normalize_ohlcv_columns

logger = logging.getLogger(__name__)


class BacktestExecutionError(Exception):
    """バックテスト実行エラー"""


class BacktestEarlyTerminationError(BacktestExecutionError):
    """戦略が意図的に早期打ち切りされたことを示す例外。"""

    def __init__(self, reason: str):
        super().__init__(f"バックテストが早期終了しました: {reason}")
        self.reason = reason


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
        strategy_parameters: Dict[str, object],
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float,
        commission_rate: float,
        slippage: float = 0.0,
        leverage: float = 1.0,
        preloaded_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        指定された戦略とパラメータでバックテスト・シミュレーションを実行します。

        このメソッドは、`backtesting.py` ライブラリのラッパーとして機能し、以下の手順を実行します：
        1. 市場データの取得: `preloaded_data` があればそれを使用し、なければ `DataService` から取得。
        2. バックテストインスタンスの生成: 証拠金、手数料、スリッページ、レバレッジ等の取引条件を設定。
        3. 戦略の実行: `strategy_class` に `strategy_parameters` を渡してシミュレーションを開始。
        4. 結果の返却: 統計データ（勝率、収益、ドローダウン等）を含むオブジェクトを返却。

        Args:
            strategy_class (Type[Strategy]): 実行する `backtesting.Strategy` 継承クラス。
            strategy_parameters (Dict[str, Any]): 指標期間やGA生成遺伝子等の戦略設定パラメータ。
            symbol (str): 取引ペア（例: "BTC/USDT"）。
            timeframe (str): 時間軸（例: "1h", "15m"）。
            start_date (datetime): 検証開始日時。
            end_date (datetime): 検証終了日時。
            initial_capital (float): 初期証拠金。
            commission_rate (float): 手数料率（0.0006 = 0.06%）。
            slippage (float): 約定コスト率（0.0001 = 0.01%）。backtesting.py の
                `spread` パラメータへ渡され、commission とは分離して扱われます。
            leverage (float): レバレッジ倍率。1.0より大きい場合、マージン率に換算して適用されます。
            preloaded_data (Optional[pd.DataFrame]): 外部でロード済みのOHLCVデータ。提供された場合はDBアクセスをスキップします。

        Returns:
            Any: パフォーマンス統計を含む `backtesting.stats` オブジェクト（Pandas Series互換）。

        Raises:
            BacktestEarlyTerminationError: 戦略内で設定された早期終了条件（大きなドローダウン等）により中断された場合。
            BacktestExecutionError: データの欠如、シミュレーション中の例外、リソース不足等により実行に失敗した場合。
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
        except BacktestEarlyTerminationError:
            raise
        except Exception as e:
            logger.error(f"バックテスト実行エラー: {e}")
            raise BacktestExecutionError(f"バックテストの実行に失敗しました: {e}")

    def _get_backtest_data(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        バックテストに使用する市場データを取得します。

        Args:
            symbol (str): 取引ペア。
            timeframe (str): 時間軸。
            start_date (datetime): 開始日時。
            end_date (datetime): 終了日時。

        Returns:
            pd.DataFrame: OHLCV形式のDataFrame。カラム名は `backtesting.py` が要求する形式に正規化されます。

        Raises:
            BacktestExecutionError: データが空の場合や取得に失敗した場合。
        """
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
            # backtesting.py が期待する OHLCV 列だけを正規化する。
            data = normalize_ohlcv_columns(data, ensure_volume=True)

            effective_commission = float(commission_rate)
            effective_spread = max(0.0, float(slippage))

            # レバレッジからマージン率を計算（例: レバレッジ10倍 -> マージン0.1）
            # 0除算防止のためmax(1.0, ...)を使用
            margin = 1.0 / max(1.0, leverage)

            bt = FractionalBacktest(
                data,
                strategy_class,
                cash=initial_capital,
                commission=effective_commission,
                spread=effective_spread,
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

    def _run_backtest(
        self, bt: Backtest, strategy_parameters: Dict[str, object]
    ) -> pd.Series:
        """バックテストを実行"""
        try:
            # 警告を一時的に無効化
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    message=r"invalid value encountered in .*",
                )
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    message=r"divide by zero encountered in .*",
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    stats = bt.run(**strategy_parameters)

            return stats
        except StrategyEarlyTermination as e:
            raise BacktestEarlyTerminationError(e.reason) from e
        except Exception as e:
            raise BacktestExecutionError(
                f"バックテスト実行中にエラーが発生しました: {e}"
            )

    def get_supported_strategies(self) -> Dict[str, object]:
        """
        サポートされている戦略一覧を取得

        Returns:
            戦略一覧
        """
        return deepcopy(SUPPORTED_STRATEGIES)
