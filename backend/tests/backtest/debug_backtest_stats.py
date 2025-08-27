#!/usr/bin/env python3
"""
バックテスト統計情報デバッグスクリプト
"""

import sys
import os

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def debug_backtest_stats():
    """バックテスト統計情報の構造をデバッグ"""
    print("=" * 60)
    print("Backtest Statistics Debug")
    print("=" * 60)

    try:
        # 簡単なテストデータでバックテストを実行
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        from backtesting import Backtest, Strategy

        # 簡単なテスト戦略
        class TestStrategy(Strategy):
            def init(self):
                pass

            def next(self):
                if len(self.data) < 2:
                    return
                # 簡単なシグナル: ランダムに取引
                if np.random.random() > 0.95:  # 5%の確率で取引
                    if not self.position:
                        self.buy()
                    else:
                        self.position.close()

        # テストデータ作成
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        dates = pd.date_range(start=start_date, end=end_date, freq='1H')

        # シンプルな価格データ
        base_price = 50000
        price_changes = np.random.normal(0, 0.01, len(dates))
        close_prices = [base_price]
        for change in price_changes[1:]:
            new_price = close_prices[-1] * (1 + change)
            close_prices.append(max(100, new_price))

        df = pd.DataFrame({
            'Open': close_prices,
            'High': [price * 1.005 for price in close_prices],
            'Low': [price * 0.995 for price in close_prices],
            'Close': close_prices,
            'Volume': np.random.uniform(1000000, 10000000, len(dates))
        }, index=dates)

        print(f"Test data created: {len(df)} rows")

        # バックテスト実行
        bt = Backtest(
            df,
            TestStrategy,
            cash=100000,
            commission=0.001,
            trade_on_close=True,
        )

        stats = bt.run()

        print("\n=== Statistics Object Analysis ===")
        print(f"Stats type: {type(stats)}")
        print(f"Stats dir: {[attr for attr in dir(stats) if not attr.startswith('_')]}")

        # 利用可能な属性を確認
        print("\n=== Available Attributes ===")
        for attr in dir(stats):
            if not attr.startswith('_'):
                try:
                    value = getattr(stats, attr)
                    print(f"{attr}: {value} (type: {type(value)})")
                except Exception as e:
                    print(f"{attr}: Error accessing - {e}")

        # 辞書アクセスを試行
        print("\n=== Dictionary Access Test ===")
        try:
            if hasattr(stats, '__getitem__') or hasattr(stats, 'get'):
                print("Dictionary-like access available")
                test_keys = ["Return [%]", "# Trades", "Win Rate [%]", "Sharpe Ratio"]
                for key in test_keys:
                    try:
                        if hasattr(stats, 'get'):
                            value = stats.get(key, 'NOT_FOUND')
                        else:
                            value = stats[key]
                        print(f"{key}: {value}")
                    except Exception as e:
                        print(f"{key}: Error - {e}")
            else:
                print("Dictionary-like access not available")
        except Exception as e:
            print(f"Dictionary access test failed: {e}")

    except Exception as e:
        print(f"Debug error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_backtest_stats()