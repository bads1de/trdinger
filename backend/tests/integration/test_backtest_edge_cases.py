"""
注意: このテストは独自実装のStrategyExecutorに依存していましたが、
backtesting.pyライブラリへの統一により無効化されました。

新しいテストは以下を参照してください:
- backend/tests/unit/test_backtest_service.py
- backend/tests/integration/test_unified_backtest_system.py
"""

import pytest

# 独自実装が削除されたため、このテストファイルは無効化
pytestmark = pytest.mark.skip(
    reason="StrategyExecutor was removed in favor of backtesting.py library"
)


import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock

# from backtest.engine.strategy_executor import StrategyExecutor
# from backtest.engine.indicators import TechnicalIndicators


@pytest.mark.integration
@pytest.mark.backtest
@pytest.mark.edge_cases
class TestBacktestEdgeCases:
    """バックテストのエッジケーステスト"""

    def test_empty_dataframe(self):
        """空のデータフレームでのテスト"""
        empty_data = pd.DataFrame()
        
        strategy_config = {
            'name': 'Test Strategy',
            'indicators': [
                {'name': 'SMA', 'params': {'period': 20}}
            ],
            'entry_rules': [
                {'condition': 'close > SMA(close, 20)'}
            ],
            'exit_rules': [
                {'condition': 'close < SMA(close, 20)'}
            ]
        }
        
        executor = StrategyExecutor()
        
        # 空データでは例外が発生するか、空の結果が返される
        with pytest.raises((IndexError, KeyError, ValueError)):
            executor.run_backtest(empty_data, strategy_config)

    def test_single_row_data(self):
        """1行のみのデータでのテスト"""
        single_row_data = pd.DataFrame({
            'Open': [100],
            'High': [110],
            'Low': [90],
            'Close': [105],
            'Volume': [1000]
        }, index=[datetime.now()])
        
        strategy_config = {
            'name': 'Single Row Test',
            'indicators': [
                {'name': 'SMA', 'params': {'period': 20}}
            ],
            'entry_rules': [
                {'condition': 'close > 100'}
            ],
            'exit_rules': [
                {'condition': 'close < 100'}
            ]
        }
        
        executor = StrategyExecutor()
        result = executor.run_backtest(single_row_data, strategy_config)
        
        # 1行のデータでは取引が発生しないはず
        assert result['total_trades'] == 0

    def test_insufficient_data_for_indicators(self):
        """指標計算に必要なデータが不足している場合"""
        # 10行のデータで20期間のSMAを計算しようとする
        insufficient_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 10),
            'High': np.random.uniform(110, 120, 10),
            'Low': np.random.uniform(90, 100, 10),
            'Close': np.random.uniform(100, 110, 10),
            'Volume': np.random.uniform(1000, 2000, 10)
        }, index=pd.date_range('2024-01-01', periods=10, freq='1H'))
        
        strategy_config = {
            'name': 'Insufficient Data Test',
            'indicators': [
                {'name': 'SMA', 'params': {'period': 20}}  # 20期間必要だが10行しかない
            ],
            'entry_rules': [
                {'condition': 'close > SMA(close, 20)'}
            ],
            'exit_rules': [
                {'condition': 'close < SMA(close, 20)'}
            ]
        }
        
        executor = StrategyExecutor()
        result = executor.run_backtest(insufficient_data, strategy_config)
        
        # データ不足でも実行は完了するが、取引は発生しない
        assert result['total_trades'] == 0

    def test_all_nan_data(self):
        """すべてNaNのデータでのテスト"""
        nan_data = pd.DataFrame({
            'Open': [np.nan] * 50,
            'High': [np.nan] * 50,
            'Low': [np.nan] * 50,
            'Close': [np.nan] * 50,
            'Volume': [np.nan] * 50
        }, index=pd.date_range('2024-01-01', periods=50, freq='1H'))
        
        strategy_config = {
            'name': 'NaN Data Test',
            'indicators': [
                {'name': 'SMA', 'params': {'period': 20}}
            ],
            'entry_rules': [
                {'condition': 'close > SMA(close, 20)'}
            ],
            'exit_rules': [
                {'condition': 'close < SMA(close, 20)'}
            ]
        }
        
        executor = StrategyExecutor()
        
        # NaNデータでは例外が発生するか、適切に処理される
        with pytest.raises((ValueError, TypeError)):
            executor.run_backtest(nan_data, strategy_config)

    def test_extreme_price_values(self):
        """極端な価格値でのテスト"""
        # 非常に大きな価格値
        extreme_data = pd.DataFrame({
            'Open': [1e10] * 50,
            'High': [1.1e10] * 50,
            'Low': [0.9e10] * 50,
            'Close': [1.05e10] * 50,
            'Volume': [1000] * 50
        }, index=pd.date_range('2024-01-01', periods=50, freq='1H'))
        
        strategy_config = {
            'name': 'Extreme Values Test',
            'indicators': [
                {'name': 'SMA', 'params': {'period': 20}}
            ],
            'entry_rules': [
                {'condition': 'close > SMA(close, 20)'}
            ],
            'exit_rules': [
                {'condition': 'close < SMA(close, 20)'}
            ]
        }
        
        executor = StrategyExecutor(initial_capital=1e15)  # 十分な資金
        result = executor.run_backtest(extreme_data, strategy_config)
        
        # 極端な値でも計算は完了する
        assert result is not None
        assert 'total_return' in result

    def test_zero_commission_rate(self):
        """手数料率ゼロでのテスト"""
        data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 50),
            'High': np.random.uniform(110, 120, 50),
            'Low': np.random.uniform(90, 100, 50),
            'Close': np.random.uniform(100, 110, 50),
            'Volume': np.random.uniform(1000, 2000, 50)
        }, index=pd.date_range('2024-01-01', periods=50, freq='1H'))
        
        strategy_config = {
            'name': 'Zero Commission Test',
            'indicators': [
                {'name': 'SMA', 'params': {'period': 10}},
                {'name': 'SMA', 'params': {'period': 20}}
            ],
            'entry_rules': [
                {'condition': 'SMA(close, 10) > SMA(close, 20)'}
            ],
            'exit_rules': [
                {'condition': 'SMA(close, 10) < SMA(close, 20)'}
            ]
        }
        
        executor = StrategyExecutor(commission_rate=0.0)
        result = executor.run_backtest(data, strategy_config)
        
        # 手数料ゼロでも正常に動作
        assert result is not None
        # 手数料がゼロなので、すべての取引で手数料は0
        for trade in result.get('trades', []):
            assert trade['commission'] == 0

    def test_maximum_commission_rate(self):
        """最大手数料率でのテスト"""
        data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 50),
            'High': np.random.uniform(110, 120, 50),
            'Low': np.random.uniform(90, 100, 50),
            'Close': np.random.uniform(100, 110, 50),
            'Volume': np.random.uniform(1000, 2000, 50)
        }, index=pd.date_range('2024-01-01', periods=50, freq='1H'))
        
        strategy_config = {
            'name': 'Max Commission Test',
            'indicators': [
                {'name': 'SMA', 'params': {'period': 10}},
                {'name': 'SMA', 'params': {'period': 20}}
            ],
            'entry_rules': [
                {'condition': 'SMA(close, 10) > SMA(close, 20)'}
            ],
            'exit_rules': [
                {'condition': 'SMA(close, 10) < SMA(close, 20)'}
            ]
        }
        
        executor = StrategyExecutor(commission_rate=1.0)  # 100%手数料
        result = executor.run_backtest(data, strategy_config)
        
        # 100%手数料でも実行は完了する（ただし利益は出ない）
        assert result is not None
        assert result['total_return'] <= 0  # 手数料で損失

    def test_invalid_date_range(self):
        """不正な日付範囲でのテスト"""
        # 逆順の日付インデックス
        dates = pd.date_range('2024-01-31', '2024-01-01', freq='-1H')
        reverse_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, len(dates)),
            'High': np.random.uniform(110, 120, len(dates)),
            'Low': np.random.uniform(90, 100, len(dates)),
            'Close': np.random.uniform(100, 110, len(dates)),
            'Volume': np.random.uniform(1000, 2000, len(dates))
        }, index=dates)
        
        strategy_config = {
            'name': 'Reverse Date Test',
            'indicators': [
                {'name': 'SMA', 'params': {'period': 20}}
            ],
            'entry_rules': [
                {'condition': 'close > SMA(close, 20)'}
            ],
            'exit_rules': [
                {'condition': 'close < SMA(close, 20)'}
            ]
        }
        
        executor = StrategyExecutor()
        result = executor.run_backtest(reverse_data, strategy_config)
        
        # 逆順でも処理は完了する
        assert result is not None

    def test_duplicate_timestamps(self):
        """重複するタイムスタンプでのテスト"""
        duplicate_dates = [datetime(2024, 1, 1)] * 50
        duplicate_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 50),
            'High': np.random.uniform(110, 120, 50),
            'Low': np.random.uniform(90, 100, 50),
            'Close': np.random.uniform(100, 110, 50),
            'Volume': np.random.uniform(1000, 2000, 50)
        }, index=duplicate_dates)
        
        strategy_config = {
            'name': 'Duplicate Timestamp Test',
            'indicators': [
                {'name': 'SMA', 'params': {'period': 20}}
            ],
            'entry_rules': [
                {'condition': 'close > SMA(close, 20)'}
            ],
            'exit_rules': [
                {'condition': 'close < SMA(close, 20)'}
            ]
        }
        
        executor = StrategyExecutor()
        result = executor.run_backtest(duplicate_data, strategy_config)
        
        # 重複タイムスタンプでも処理は完了する
        assert result is not None

    def test_missing_required_columns(self):
        """必要な列が欠けているデータでのテスト"""
        incomplete_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 50),
            'High': np.random.uniform(110, 120, 50),
            # 'Low'と'Close'が欠けている
            'Volume': np.random.uniform(1000, 2000, 50)
        }, index=pd.date_range('2024-01-01', periods=50, freq='1H'))
        
        strategy_config = {
            'name': 'Missing Columns Test',
            'indicators': [
                {'name': 'SMA', 'params': {'period': 20}}
            ],
            'entry_rules': [
                {'condition': 'close > SMA(close, 20)'}
            ],
            'exit_rules': [
                {'condition': 'close < SMA(close, 20)'}
            ]
        }
        
        executor = StrategyExecutor()
        
        # 必要な列が欠けている場合は例外が発生
        with pytest.raises((KeyError, ValueError)):
            executor.run_backtest(incomplete_data, strategy_config)

    def test_extremely_volatile_market(self):
        """極端に変動の激しい市場でのテスト"""
        # 価格が激しく上下する市場を模擬
        volatile_prices = []
        base_price = 100
        for i in range(100):
            # 50%の確率で±20%の変動
            if np.random.random() > 0.5:
                change = np.random.uniform(0.8, 1.2)
            else:
                change = np.random.uniform(0.8, 1.2)
            base_price *= change
            volatile_prices.append(base_price)
        
        volatile_data = pd.DataFrame({
            'Open': volatile_prices,
            'High': [p * 1.1 for p in volatile_prices],
            'Low': [p * 0.9 for p in volatile_prices],
            'Close': volatile_prices,
            'Volume': np.random.uniform(1000, 2000, 100)
        }, index=pd.date_range('2024-01-01', periods=100, freq='1H'))
        
        strategy_config = {
            'name': 'Volatile Market Test',
            'indicators': [
                {'name': 'SMA', 'params': {'period': 10}},
                {'name': 'RSI', 'params': {'period': 14}}
            ],
            'entry_rules': [
                {'condition': 'RSI(close, 14) < 30'}
            ],
            'exit_rules': [
                {'condition': 'RSI(close, 14) > 70'}
            ]
        }
        
        executor = StrategyExecutor()
        result = executor.run_backtest(volatile_data, strategy_config)
        
        # 極端に変動の激しい市場でも処理は完了する
        assert result is not None
        assert 'total_return' in result

    def test_flat_market_no_volatility(self):
        """変動のない平坦な市場でのテスト"""
        # すべて同じ価格
        flat_price = 100
        flat_data = pd.DataFrame({
            'Open': [flat_price] * 100,
            'High': [flat_price] * 100,
            'Low': [flat_price] * 100,
            'Close': [flat_price] * 100,
            'Volume': [1000] * 100
        }, index=pd.date_range('2024-01-01', periods=100, freq='1H'))
        
        strategy_config = {
            'name': 'Flat Market Test',
            'indicators': [
                {'name': 'SMA', 'params': {'period': 20}},
                {'name': 'RSI', 'params': {'period': 14}}
            ],
            'entry_rules': [
                {'condition': 'close > SMA(close, 20)'}
            ],
            'exit_rules': [
                {'condition': 'close < SMA(close, 20)'}
            ]
        }
        
        executor = StrategyExecutor()
        result = executor.run_backtest(flat_data, strategy_config)
        
        # 変動がない市場では取引が発生しない
        assert result is not None
        assert result['total_trades'] == 0
        assert result['total_return'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
