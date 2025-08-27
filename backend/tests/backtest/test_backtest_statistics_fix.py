#!/usr/bin/env python3
"""
バックテスト統計情報修正テスト
"""

import sys
import os

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_backtest_statistics_fix():
    """バックテスト統計情報の修正をテスト"""
    print("=" * 60)
    print("Backtest Statistics Fix Test")
    print("=" * 60)

    try:
        from app.services.backtest.backtest_service import BacktestService
        from database.connection import SessionLocal
        from database.repositories.backtest_result_repository import BacktestResultRepository

        # 簡単なテスト戦略でバックテストを実行
        from backtesting import Strategy
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta

        class SimpleTestStrategy(Strategy):
            def init(self):
                pass

            def next(self):
                if len(self.data) < 2:
                    return
                # 簡単なシグナル: 5%の確率でランダム取引
                if np.random.random() > 0.95:
                    if not self.position:
                        self.buy()
                    else:
                        self.position.close()

        # テストデータ作成
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        dates = pd.date_range(start=start_date, end=end_date, freq='1H')

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

        # 直接BacktestExecutorを使って実行（データベースを経由しない）
        from app.services.backtest.execution.backtest_executor import BacktestExecutor
        from app.services.backtest.backtest_data_service import BacktestDataService

        # モックデータサービスを作成
        class MockDataService:
            def get_data_for_backtest(self, symbol, timeframe, start_date, end_date):
                return df

        # BacktestExecutorで直接実行
        data_service = MockDataService()
        executor = BacktestExecutor(data_service)

        print("Running backtest with fixed statistics extraction...")
        stats = executor.execute_backtest(
            strategy_class=SimpleTestStrategy,
            strategy_parameters={},
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            start_date=df.index[0],
            end_date=df.index[-1],
            initial_capital=100000.0,
            commission_rate=0.001
        )

        # 結果変換
        from app.services.backtest.conversion.backtest_result_converter import BacktestResultConverter
        converter = BacktestResultConverter()

        result = converter.convert_backtest_results(
            stats=stats,
            strategy_name="Test_Statistics_Fix",
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            initial_capital=100000.0,
            start_date=df.index[0],
            end_date=df.index[-1],
            config_json={"test": "statistics_fix"}
        )

        print("\n=== Backtest Result Analysis ===")
        print(f"Strategy Name: {result.get('strategy_name')}")
        print(f"Symbol: {result.get('symbol')}")
        print(f"Timeframe: {result.get('timeframe')}")

        # 統計情報確認
        performance_metrics = result.get('performance_metrics', {})
        print("\nPerformance Metrics:")
        for key, value in performance_metrics.items():
            print(f"  {key}: {value}")

        # 重要な指標が0でないことを確認
        critical_metrics = ['total_return', 'total_trades', 'sharpe_ratio']
        zero_metrics = []
        non_zero_metrics = []

        for metric in critical_metrics:
            if metric in performance_metrics:
                value = performance_metrics[metric]
                if value == 0 or value == 0.0:
                    zero_metrics.append(metric)
                else:
                    non_zero_metrics.append(f"{metric}={value}")

        print(f"\nZero metrics: {zero_metrics}")
        print(f"Non-zero metrics: {non_zero_metrics}")

        if len(zero_metrics) == 0:
            print("\n[SUCCESS] All critical metrics have non-zero values!")
            return True
        else:
            print(f"\n[WARNING] {len(zero_metrics)} metrics are still zero: {zero_metrics}")
            return False

    except Exception as e:
        print(f"Error in backtest statistics fix test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_backtest_statistics_fix()
    print(f"\nFinal result: {'PASS' if success else 'FAIL'}")
    sys.exit(0 if success else 1)