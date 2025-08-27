#!/usr/bin/env python3
"""
シンプルなバックテストデバッグ
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# ログレベルを設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def simple_backtest_debug():
    """シンプルなバックテストでデバッグ"""
    print("=" * 60)
    print("Simple Backtest Debug")
    print("=" * 60)

    try:
        from backtesting import Backtest, Strategy
        import pandas as pd
        import numpy as np
        from app.services.backtest.conversion.backtest_result_converter import (
            BacktestResultConverter,
        )

        # シンプルなテストデータを作成
        dates = pd.date_range("2024-01-01", periods=100, freq="H")
        np.random.seed(42)

        # ランダムウォークでOHLCVデータを生成
        close_prices = 50000 + np.cumsum(np.random.randn(100) * 100)
        high_prices = close_prices + np.random.rand(100) * 200
        low_prices = close_prices - np.random.rand(100) * 200
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        volume = np.random.randint(1000, 10000, 100)

        data = pd.DataFrame(
            {
                "Open": open_prices,
                "High": high_prices,
                "Low": low_prices,
                "Close": close_prices,
                "Volume": volume,
            },
            index=dates,
        )

        print(f"テストデータ作成完了: {len(data)}行")
        print(f"価格範囲: {data['Close'].min():.2f} - {data['Close'].max():.2f}")

        # シンプルな戦略クラス
        class SimpleTestStrategy(Strategy):
            def init(self):
                # シンプルなSMA
                self.sma20 = self.I(
                    lambda x: pd.Series(x).rolling(20).mean(), self.data.Close
                )

            def next(self):
                # シンプルなクロスオーバー戦略
                if not self.position:
                    if (
                        self.data.Close[-1] > self.sma20[-1]
                        and self.data.Close[-2] <= self.sma20[-2]
                    ):
                        # 非常に小さなサイズで取引
                        self.buy(size=0.01)  # 0.01 BTC
                elif self.position:
                    if (
                        self.data.Close[-1] < self.sma20[-1]
                        and self.data.Close[-2] >= self.sma20[-2]
                    ):
                        self.position.close()

        # バックテスト実行
        print("バックテスト実行中...")
        bt = Backtest(data, SimpleTestStrategy, cash=100000, commission=0.001)
        stats = bt.run()

        print("バックテスト完了!")
        print(f"statsの型: {type(stats)}")
        print(
            f"statsの属性: {[attr for attr in dir(stats) if not attr.startswith('_')]}"
        )

        # 基本統計情報を表示
        print(f"\n基本統計情報:")
        print(f"  Return [%]: {stats.get('Return [%]', 'N/A')}")
        print(f"  # Trades: {stats.get('# Trades', 'N/A')}")
        print(f"  Win Rate [%]: {stats.get('Win Rate [%]', 'N/A')}")
        print(f"  Profit Factor: {stats.get('Profit Factor', 'N/A')}")
        print(f"  Sharpe Ratio: {stats.get('Sharpe Ratio', 'N/A')}")

        # _trades属性を確認
        print(f"\n_trades属性確認:")
        if hasattr(stats, "_trades"):
            trades_df = getattr(stats, "_trades")
            print(f"  _tradesの型: {type(trades_df)}")
            if trades_df is not None:
                print(f"  _tradesの長さ: {len(trades_df)}")
                if hasattr(trades_df, "columns"):
                    print(f"  _tradesの列: {list(trades_df.columns)}")
                if len(trades_df) > 0:
                    print(f"  最初の取引:")
                    for col in trades_df.columns:
                        print(f"    {col}: {trades_df.iloc[0][col]}")
        else:
            print("  _trades属性が存在しません")

        # コンバーターでテスト
        print(f"\nコンバーターテスト:")
        converter = BacktestResultConverter()

        config_json = {"commission_rate": 0.001}
        result = converter.convert_backtest_results(
            stats=stats,
            strategy_name="Simple_Test_Strategy",
            symbol="BTC/USDT",
            timeframe="1h",
            initial_capital=100000.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 5),
            config_json=config_json,
        )

        print(f"変換結果:")
        metrics = result.get("performance_metrics", {})
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        trade_history = result.get("trade_history", [])
        print(f"\n取引履歴: {len(trade_history)}件")

    except Exception as e:
        print(f"エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    simple_backtest_debug()
