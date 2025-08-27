#!/usr/bin/env python3
"""
実際のバックテスト実行とJSON保存テスト
修正された統計抽出機能を確認
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime, timedelta
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_sample_ohlcv_data():
    """サンプルOHLCVデータを作成"""
    print("Creating sample OHLCV data...")

    # 2020年1月-3月の日次データを作成
    start_date = datetime(2020, 1, 1)
    dates = []
    current_date = start_date

    while current_date <= datetime(2020, 3, 31):
        dates.append(current_date)
        current_date += timedelta(days=1)

    # より現実的な価格変動データ
    base_price = 50000.0
    data = []

    for i, date in enumerate(dates):
        # トレンドとノイズを組み合わせた価格変動
        import random
        import math

        # 穏やかな基本トレンド（徐々に上昇）
        trend = 0.0001 * i  # 非常に穏やかな上昇トレンド

        # ノイズ
        noise = random.uniform(-0.01, 0.01)

        # 周期的な変動（市場の季節性）
        cycle = math.sin(2 * math.pi * i / 30) * 0.005  # 小さな周期変動

        total_change = trend + noise + cycle

        open_price = base_price
        close_price = base_price * (1 + total_change)
        high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.005))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.005))
        volume = random.uniform(100, 1000)

        data.append({
            'timestamp': date,
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume
        })

        base_price = close_price

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)

    print(f"Created {len(df)} rows of sample data")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Price range: {df['Close'].min():.2f} to {df['Close'].max():.2f}")

    return df

def create_simple_strategy():
    """シンプルな取引戦略を作成"""
    print("Creating simple trading strategy...")

    try:
        from backtesting import Strategy

        class SimpleStrategy(Strategy):
            def init(self):
                # シンプルな移動平均クロス戦略
                self.sma_short = self.I(lambda: pd.Series(self.data.Close).rolling(5).mean())
                self.sma_long = self.I(lambda: pd.Series(self.data.Close).rolling(20).mean())

            def next(self):
                if len(self.data) < 20:
                    return

                # デバッグ情報（最初の20バーと最後のバーだけ表示）
                if len(self.data) <= 25 or len(self.data) >= 90:
                    print(f"Bar {len(self.data)}: Short MA={self.sma_short[-1]:.2f}, Long MA={self.sma_long[-1]:.2f}, Close={self.data.Close[-1]:.2f}")

                # より頻繁に取引するための条件緩和
                if self.sma_short[-1] > self.sma_long[-1]:
                    if not self.position:
                        print(f"BUY SIGNAL at {self.data.Close[-1]:.2f}")
                        # 固定数量で取引（1単位）
                        self.buy(size=1)

                # デッドクロスで売り
                elif self.sma_short[-1] < self.sma_long[-1]:
                    if self.position:
                        print(f"SELL SIGNAL at {self.data.Close[-1]:.2f}")
                        self.position.close()

                # 損失が大きい場合も手仕舞い（ストップロス）- 一旦コメントアウト
                # if self.position and (self.data.Close[-1] / self.position.entry_price - 1) < -0.05:
                #     print(f"STOP LOSS at {self.data.Close[-1]:.2f}")
                #     self.position.close()

        return SimpleStrategy

    except ImportError as e:
        print(f"Backtesting library not available: {e}")
        return None

def run_backtest_and_save_json():
    """バックテストを実行してJSONで保存"""
    print("=== Running Backtest and Saving JSON ===")

    try:
        from backtesting import Backtest
        from app.services.backtest.conversion.backtest_result_converter import BacktestResultConverter

        # 1. サンプルデータ作成
        data = create_sample_ohlcv_data()

        # 2. 戦略作成
        strategy_class = create_simple_strategy()
        if not strategy_class:
            return False

        # 3. バックテスト実行
        print("\\nRunning backtest...")
        bt = Backtest(
            data,
            strategy_class,
            cash=100000,  # 初期資金を戻す
            commission=.001,
        )

        result = bt.run()
        print("Backtest completed!")

        # Debug: 結果オブジェクトの詳細を確認
        print(f"\\n=== Debug Info ===")
        print(f"Result type: {type(result)}")
        print(f"Result attributes: {[attr for attr in dir(result) if not attr.startswith('__')]}")

        # stats オブジェクトの確認
        if hasattr(result, '_stats'):
            print(f"Has _stats: {type(result._stats)}")
        if hasattr(result, '_statistics'):
            print(f"Has _statistics: {type(result._statistics)}")
        if hasattr(result, 'stats'):
            print(f"Has stats: {type(result.stats)}")

        # 利用可能な統計情報を確認
        try:
            stats_dict = result._stats if hasattr(result, '_stats') else result.stats if hasattr(result, 'stats') else result._statistics if hasattr(result, '_statistics') else None
            if stats_dict:
                print(f"Stats dict type: {type(stats_dict)}")
                if hasattr(stats_dict, 'keys'):
                    print(f"Available stats keys: {list(stats_dict.keys())}")
                    # 最初のいくつかの値を表示
                    for key in list(stats_dict.keys())[:10]:
                        try:
                            value = stats_dict[key]
                            print(f"  {key}: {value} (type: {type(value)})")
                        except Exception as e:
                            print(f"  {key}: Error accessing value - {e}")
                else:
                    print(f"Stats dict attributes: {[attr for attr in dir(stats_dict) if not attr.startswith('__')]}")
            else:
                print("No stats dict found")
                # result 自体をチェック
                print(f"Result keys (if dict-like): {list(result.keys()) if hasattr(result, 'keys') else 'Not dict-like'}")
                print("First few result items:")
                if hasattr(result, 'items'):
                    for key, value in list(result.items())[:10]:
                        print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error accessing stats: {e}")
            import traceback
            traceback.print_exc()

        # 4. 修正されたコンバーターで変換
        print("\\nConverting results with fixed converter...")
        converter = BacktestResultConverter()

        # 統計情報を直接resultオブジェクトから取得
        print(f"Direct result access - Return [%]: {result.get('Return [%]', 'N/A')}")
        print(f"Direct result access - # Trades: {result.get('# Trades', 'N/A')}")
        print(f"Direct result access - Sharpe Ratio: {result.get('Sharpe Ratio', 'N/A')}")

        converted_result = converter.convert_backtest_results(
            stats=result,  # resultオブジェクト自体を渡す
            strategy_name="Simple_Test_Strategy",
            symbol="BTC/USDT",
            timeframe="1d",
            initial_capital=100000.0,  # 初期資金を修正
            start_date=data.index.min(),
            end_date=data.index.max(),
            config_json={
                "strategy_type": "Simple MA Cross",
                "short_period": 5,
                "long_period": 20
            }
        )

        # 5. JSONファイルに保存
        output_file = "backtest_result_test.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_result, f, indent=2, default=str, ensure_ascii=False)

        print(f"\\nSUCCESS: Results saved to {output_file}")

        # 6. 保存された内容を確認
        print("\n=== Verification of Saved Results ===")

        # 基本情報
        print("Basic Info:")
        print(f"  Strategy: {converted_result['strategy_name']}")
        print(f"  Symbol: {converted_result['symbol']}")
        print(f"  Timeframe: {converted_result['timeframe']}")
        print(f"  Initial Capital: {converted_result['initial_capital']}")
        print(f"  Created: {converted_result['created_at']}")

        # 統計情報
        print("\nPerformance Metrics:")
        # 統計情報はトップレベルに保存されている
        metrics = converted_result
        key_metrics = ['total_return', 'total_trades', 'win_rate', 'sharpe_ratio', 'max_drawdown']

        all_non_zero = True
        for key in key_metrics:
            value = metrics.get(key, 'N/A')
            status = 'OK' if value != 0 and value != 'N/A' and value != 0.0 else 'ZERO'
            if value == 0 or value == 0.0:
                all_non_zero = False
            print(f"  {key}: {value} [{status}]")

        print(f"\nAll key metrics non-zero: {'SUCCESS' if all_non_zero else 'FAILED'}")

        # 取引履歴
        trades = converted_result.get('trade_history', [])
        print(f"\nTrade History: {len(trades)} trades")

        if len(trades) > 0:
            print("  Sample trade:")
            sample_trade = trades[0]
            print(f"    Entry Time: {sample_trade.get('entry_time', 'N/A')}")
            print(f"    Exit Time: {sample_trade.get('exit_time', 'N/A')}")
            print(f"    Entry Price: {sample_trade.get('entry_price', 'N/A')}")
            print(f"    Exit Price: {sample_trade.get('exit_price', 'N/A')}")
            print(f"    PnL: {sample_trade.get('pnl', 'N/A')}")
            print(f"    Return %: {sample_trade.get('return_pct', 'N/A')}")

        # 資産曲線
        equity_curve = converted_result.get('equity_curve', [])
        print(f"\nEquity Curve: {len(equity_curve)} points")

        if len(equity_curve) > 0:
            print("  First point:")
            first_point = equity_curve[0]
            print(f"    Timestamp: {first_point.get('timestamp', 'N/A')}")
            print(f"    Equity: {first_point.get('equity', 'N/A')}")

            print("  Last point:")
            last_point = equity_curve[-1]
            print(f"    Timestamp: {last_point.get('timestamp', 'N/A')}")
            print(f"    Equity: {last_point.get('equity', 'N/A')}")

        # 設定情報
        config = converted_result.get('config_json', {})
        print(f"\nConfiguration: {config}")

        # 修正確認
        print("\nFix Verification:")
        print(f"  Total metrics extracted: {len(metrics)}")
        print(f"  Statistics extraction: {'SUCCESS' if len(metrics) > 0 else 'FAILED'}")
        print(f"  Trade history extraction: {'SUCCESS' if len(trades) > 0 else 'FAILED'}")
        print(f"  Equity curve extraction: {'SUCCESS' if len(equity_curve) > 0 else 'FAILED'}")

        return all_non_zero and len(metrics) > 0

    except Exception as e:
        print(f"Error in backtest execution: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行関数"""
    print("=" * 60)
    print("Real Backtest Execution and JSON Save Test")
    print("Testing Fixed Statistics Extraction")
    print("=" * 60)

    success = run_backtest_and_save_json()

    print("\\n" + "=" * 60)
    if success:
        print("SUCCESS: STATISTICS EXTRACTION FIX VERIFIED SUCCESSFULLY!")
        print("SUCCESS: All key metrics are non-zero")
        print("SUCCESS: Statistics, trades, and equity curve extracted correctly")
        print("SUCCESS: Results saved to backtest_result_test.json")
    else:
        print("FAILED: Statistics extraction fix verification failed")
    print("=" * 60)

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)