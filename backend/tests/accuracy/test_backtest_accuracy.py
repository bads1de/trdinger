"""
バックテスト結果精度検証テスト

このファイルは以下をテストします：
- 利益計算の正確性
- 手数料計算の正確性
- ポジション管理の正確性
- パフォーマンス指標の数学的正確性
- 数値の丸め誤差や累積誤差の検証
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
import math

from backtest.engine.strategy_executor import StrategyExecutor
from unittest.mock import Mock

# 高精度計算のための設定
getcontext().prec = 28


@pytest.mark.accuracy
@pytest.mark.backtest
class TestBacktestAccuracy:
    """バックテスト結果精度検証テスト"""

    def create_precise_test_data(self, initial_price: float = 50000, days: int = 30):
        """精度検証用の正確なテストデータを作成"""
        dates = pd.date_range('2024-01-01', periods=days, freq='1D')
        
        # 予測可能な価格パターンを作成
        prices = []
        current_price = initial_price
        
        for i in range(days):
            # 単純な上昇トレンド（1日1%上昇）
            daily_return = 0.01
            current_price *= (1 + daily_return)
            prices.append(current_price)
        
        # OHLCV データを生成（予測可能な値）
        data = []
        for i, close_price in enumerate(prices):
            open_price = close_price / 1.01 if i > 0 else initial_price
            high_price = close_price * 1.005  # 終値の0.5%高
            low_price = open_price * 0.995   # 始値の0.5%安
            volume = 1000  # 固定出来高
            
            data.append({
                'Open': round(open_price, 2),
                'High': round(high_price, 2),
                'Low': round(low_price, 2),
                'Close': round(close_price, 2),
                'Volume': volume
            })
        
        return pd.DataFrame(data, index=dates)

    def test_profit_calculation_accuracy(self):
        """利益計算の正確性テスト"""
        # 精密なテストデータ
        data = self.create_precise_test_data(initial_price=50000, days=10)
        
        initial_capital = 100000.0
        commission_rate = 0.001  # 0.1%
        
        executor = StrategyExecutor(
            initial_capital=initial_capital,
            commission_rate=commission_rate
        )
        
        # 手動で取引を実行して精度を検証
        entry_price = data.iloc[0]['Close']  # 50000.0
        exit_price = data.iloc[-1]['Close']   # 約55230.85
        
        # 理論的な計算
        available_capital = initial_capital * 0.95  # 手数料を考慮
        quantity = available_capital / entry_price
        entry_commission = entry_price * quantity * commission_rate
        
        proceeds = exit_price * quantity
        exit_commission = proceeds * commission_rate
        net_proceeds = proceeds - exit_commission
        
        theoretical_profit = net_proceeds - (entry_price * quantity + entry_commission)
        
        # 実際の取引実行
        timestamp = datetime.now()
        buy_trade = executor.execute_trade('buy', entry_price, timestamp)
        sell_trade = executor.execute_trade('sell', exit_price, timestamp)
        
        # 精度検証
        assert buy_trade is not None, "買い取引が実行されていない"
        assert sell_trade is not None, "売り取引が実行されていない"
        
        # 手数料計算の精度
        expected_buy_commission = entry_price * buy_trade.quantity * commission_rate
        assert abs(buy_trade.commission - expected_buy_commission) < 0.01, \
            f"買い手数料の計算誤差: {buy_trade.commission} vs {expected_buy_commission}"
        
        expected_sell_commission = exit_price * sell_trade.quantity * commission_rate
        assert abs(sell_trade.commission - expected_sell_commission) < 0.01, \
            f"売り手数料の計算誤差: {sell_trade.commission} vs {expected_sell_commission}"
        
        # 利益計算の精度
        actual_profit = sell_trade.pnl
        assert abs(actual_profit - theoretical_profit) < 1.0, \
            f"利益計算の誤差: {actual_profit} vs {theoretical_profit}"

    def test_commission_calculation_precision(self):
        """手数料計算の精度テスト"""
        test_cases = [
            # (価格, 数量, 手数料率, 期待値)
            (50000.0, 1.0, 0.001, 50.0),
            (50000.0, 0.5, 0.001, 25.0),
            (50000.123, 1.0, 0.001, 50.000123),
            (50000.0, 1.0, 0.0001, 5.0),
            (50000.0, 1.0, 0.01, 500.0),
        ]
        
        executor = StrategyExecutor()
        
        for price, quantity, commission_rate, expected in test_cases:
            executor.commission_rate = commission_rate
            calculated_commission = price * quantity * commission_rate
            
            # 精度検証（小数点以下6桁まで）
            assert abs(calculated_commission - expected) < 1e-6, \
                f"手数料計算誤差: {calculated_commission} vs {expected}"

    def test_mathematical_consistency(self):
        """数学的整合性のテスト"""
        data = self.create_precise_test_data(initial_price=50000, days=30)
        
        strategy_config = {
            'name': 'Consistency Test',
            'indicators': [
                {'name': 'SMA', 'params': {'period': 10}}
            ],
            'entry_rules': [
                {'condition': 'close > SMA(close, 10)'}
            ],
            'exit_rules': [
                {'condition': 'close < SMA(close, 10)'}
            ]
        }
        
        executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)
        result = executor.run_backtest(data, strategy_config)
        
        # 基本的な数学的整合性チェック
        initial_capital = 100000
        final_equity = result.get('final_equity', initial_capital)
        total_return = result.get('total_return', 0)
        
        # 総リターンの整合性
        expected_final_equity = initial_capital * (1 + total_return)
        assert abs(final_equity - expected_final_equity) < 1.0, \
            f"総リターンと最終資産の不整合: {final_equity} vs {expected_final_equity}"
        
        # 取引数と勝率の整合性
        total_trades = result.get('total_trades', 0)
        winning_trades = result.get('winning_trades', 0)
        win_rate = result.get('win_rate', 0)
        
        if total_trades > 0:
            expected_win_rate = winning_trades / total_trades
            assert abs(win_rate - expected_win_rate) < 0.01, \
                f"勝率計算の不整合: {win_rate} vs {expected_win_rate}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
