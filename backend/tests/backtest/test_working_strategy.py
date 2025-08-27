#!/usr/bin/env python3
"""
確実に取引が実行される戦略でテスト
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


def test_working_strategy():
    """確実に取引が実行される戦略でテスト"""
    print("=" * 60)
    print("Working Strategy Test")
    print("=" * 60)

    try:
        from backtesting import Backtest, Strategy
        import pandas as pd
        import numpy as np
        from app.services.backtest.conversion.backtest_result_converter import (
            BacktestResultConverter,
        )

        # より長いテストデータを作成（取引機会を増やす）
        dates = pd.date_range("2024-01-01", periods=500, freq="h")
        np.random.seed(42)

        # より変動の大きいランダムウォークでOHLCVデータを生成
        price_changes = np.random.randn(500) * 500  # より大きな変動
        close_prices = 50000 + np.cumsum(price_changes)
        high_prices = close_prices + np.random.rand(500) * 300
        low_prices = close_prices - np.random.rand(500) * 300
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        volume = np.random.randint(1000, 10000, 500)

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

        # 確実に取引が実行される戦略クラス
        class WorkingStrategy(Strategy):
            def init(self):
                # シンプルなSMA
                self.sma10 = self.I(
                    lambda x: pd.Series(x).rolling(10).mean(), self.data.Close
                )
                self.sma20 = self.I(
                    lambda x: pd.Series(x).rolling(20).mean(), self.data.Close
                )

            def next(self):
                # より頻繁に取引するシンプルな戦略
                if not self.position:
                    # ゴールデンクロス（短期SMAが長期SMAを上抜け）
                    if (
                        len(self.data) > 20
                        and self.sma10[-1] > self.sma20[-1]
                        and self.sma10[-2] <= self.sma20[-2]
                    ):
                        # 非常に小さなサイズで買い（証拠金不足を避ける）
                        self.buy(size=0.001)  # 0.001 BTC
                elif self.position:
                    # デッドクロス（短期SMAが長期SMAを下抜け）
                    if (
                        self.sma10[-1] < self.sma20[-1]
                        and self.sma10[-2] >= self.sma20[-2]
                    ):
                        self.position.close()

        # バックテスト実行（より大きな初期資金）
        print("バックテスト実行中...")
        bt = Backtest(
            data, WorkingStrategy, cash=1000000, commission=0.001
        )  # 100万ドル
        stats = bt.run()

        print("バックテスト完了!")
        print(f"statsの型: {type(stats)}")

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
                    print(f"  最後の取引:")
                    for col in trades_df.columns:
                        print(f"    {col}: {trades_df.iloc[-1][col]}")
        else:
            print("  _trades属性が存在しません")

        # コンバーターでテスト
        print(f"\nコンバーターテスト:")
        converter = BacktestResultConverter()

        config_json = {"commission_rate": 0.001}
        result = converter.convert_backtest_results(
            stats=stats,
            strategy_name="Working_Test_Strategy",
            symbol="BTC/USDT",
            timeframe="1h",
            initial_capital=1000000.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 21),
            config_json=config_json,
        )

        print(f"変換結果:")
        metrics = result.get("performance_metrics", {})
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        trade_history = result.get("trade_history", [])
        print(f"\n取引履歴: {len(trade_history)}件")
        if trade_history:
            print("最初の取引:")
            for key, value in trade_history[0].items():
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_working_strategy()
