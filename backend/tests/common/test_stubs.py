"""
共通テスト用スタブクラス

このモジュールには、複数のテストファイルで重複しているスタブクラスを統合しています。
"""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


class SyntheticDataService:
    """バックテスト用の合成データを提供するサービス"""

    def __init__(self, data_generator_func=None):
        """
        初期化

        Args:
            data_generator_func: データ生成関数（オプション）
        """
        self.data_generator_func = data_generator_func

    def get_data_for_backtest(self, symbol, timeframe, start_date, end_date):
        """バックテスト用データを取得"""
        if self.data_generator_func:
            return self.data_generator_func(start_date, end_date)
        else:
            # デフォルトの実装
            bars = int((end_date - start_date).total_seconds() // 3600) + 1
            return self._make_hourly_data(start_date, bars)

    def _make_hourly_data(self, start: datetime, bars: int) -> pd.DataFrame:
        """1時間足の合成価格データを作成"""
        idx = pd.date_range(start=start, periods=bars, freq="1h")
        t = np.linspace(0, 20 * np.pi, bars)
        trend = np.linspace(100, 180, bars)
        wavy = 8 * np.sin(t) + 4 * np.sin(3.1 * t)
        noise = np.random.normal(0, 0.8, size=bars)
        close = trend + wavy + noise
        open_ = np.concatenate([[close[0]], close[:-1]])
        high = np.maximum(open_, close) * (1 + np.random.normal(0.0008, 0.0005, size=bars))
        low = np.minimum(open_, close) * (1 - np.random.normal(0.0008, 0.0005, size=bars))
        vol = np.full(bars, 2000.0)
        return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


class SyntheticBacktestDataService:
    """バックテスト用の合成データを提供する軽量スタブ"""

    def __init__(self, n: int = 300, seed: int = 0, base_price: float = 50000.0):
        """
        初期化

        Args:
            n: データポイント数
            seed: 乱数シード
            base_price: 基準価格
        """
        self.n = n
        self.seed = seed
        self.base_price = base_price

    def get_data_for_backtest(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """バックテスト用データを取得"""
        rng = np.random.default_rng(self.seed)
        idx = pd.date_range(start=start_date, end=end_date, periods=self.n)
        base = self.base_price
        # ランダムウォークにボラティリティを加味
        returns = rng.normal(0, 0.01, size=self.n)
        prices = [base]
        for r in returns[1:]:
            prices.append(max(1000.0, prices[-1] * (1 + r)))
        close = np.array(prices)
        high = close * (1 + np.abs(rng.normal(0, 0.004, size=self.n)))
        low = close * (1 - np.abs(rng.normal(0, 0.004, size=self.n)))
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        vol = rng.integers(100, 10000, size=self.n)
        df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)
        return df


class DummyBacktestService:
    """ダミーバックテストサービス"""

    def __init__(self, total_return: float = 0.1, sharpe_ratio: float = 1.2,
                 max_drawdown: float = 0.2, win_rate: float = 0.55, total_trades: int = 20):
        """
        初期化

        Args:
            total_return: 総リターン
            sharpe_ratio: シャープレシオ
            max_drawdown: 最大ドローダウン
            win_rate: 勝率
            total_trades: 総トレード数
        """
        self.total_return = total_return
        self.sharpe_ratio = sharpe_ratio
        self.max_drawdown = max_drawdown
        self.win_rate = win_rate
        self.total_trades = total_trades

    def run_backtest(self, config):
        """バックテストを実行"""
        return {
            "performance_metrics": {
                "total_return": self.total_return,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown,
                "win_rate": self.win_rate,
                "total_trades": self.total_trades,
            },
            "trade_history": [],
        }


# 後方互換性のためのエイリアス
_SyntheticDataService = SyntheticDataService
_SyntheticBacktestDataService = SyntheticBacktestDataService
DummyBacktestService = DummyBacktestService