"""
戦略実行エンジンのテスト
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtest.engine.strategy_executor import StrategyExecutor, Trade, Position


class TestStrategyExecutor:
    """戦略実行エンジンのテストクラス"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータを生成"""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')

        # 上昇トレンドのデータを生成（テストしやすくするため）
        base_price = 100
        trend = np.linspace(0, 20, 50)  # 20ポイントの上昇
        noise = np.random.RandomState(42).randn(50) * 0.5
        prices = base_price + trend + noise

        data = pd.DataFrame({
            'open': prices + np.random.RandomState(42).randn(50) * 0.1,
            'high': prices + np.abs(np.random.RandomState(42).randn(50) * 0.3),
            'low': prices - np.abs(np.random.RandomState(42).randn(50) * 0.3),
            'close': prices,
            'volume': np.random.RandomState(42).randint(1000, 10000, 50)
        }, index=dates)

        return data

    @pytest.fixture
    def executor(self):
        """戦略実行エンジンのインスタンスを作成"""
        return StrategyExecutor(initial_capital=100000, commission_rate=0.001)

    def test_initialization(self, executor):
        """初期化のテスト"""
        assert executor.initial_capital == 100000
        assert executor.commission_rate == 0.001
        assert executor.capital == 100000
        assert executor.position.quantity == 0
        assert executor.position.avg_price == 0
        assert len(executor.trades) == 0
        assert len(executor.equity_curve) == 0

    def test_reset(self, executor):
        """リセット機能のテスト"""
        # 状態を変更
        executor.capital = 50000
        executor.position.quantity = 10
        executor.trades.append(Trade(datetime.now(), 'buy', 100, 10, 1))

        # リセット
        executor.reset()

        # 初期状態に戻っているか確認
        assert executor.capital == 100000
        assert executor.position.quantity == 0
        assert len(executor.trades) == 0
        assert len(executor.equity_curve) == 0

    def test_calculate_indicators(self, executor, sample_data):
        """指標計算のテスト"""
        indicators_config = [
            {'name': 'SMA', 'params': {'period': 10}},
            {'name': 'RSI', 'params': {'period': 14}}
        ]

        indicators = executor.calculate_indicators(sample_data, indicators_config)

        assert 'SMA_10' in indicators
        assert 'RSI_14' in indicators
        assert len(indicators['SMA_10']) == len(sample_data)
        assert len(indicators['RSI_14']) == len(sample_data)

    def test_execute_buy_trade(self, executor):
        """買い取引のテスト"""
        price = 100
        timestamp = datetime.now()

        trade = executor.execute_trade('buy', price, timestamp)

        assert trade is not None
        assert trade.type == 'buy'
        assert trade.price == price
        assert executor.position.quantity > 0
        assert executor.capital < executor.initial_capital

    def test_execute_sell_trade(self, executor):
        """売り取引のテスト"""
        # まず買いポジションを作成
        buy_price = 100
        buy_timestamp = datetime.now()
        executor.execute_trade('buy', buy_price, buy_timestamp)

        initial_position = executor.position.quantity

        # 売り取引
        sell_price = 110
        sell_timestamp = datetime.now()
        trade = executor.execute_trade('sell', sell_price, sell_timestamp)

        assert trade is not None
        assert trade.type == 'sell'
        assert trade.price == sell_price
        assert trade.pnl > 0  # 利益が出ているはず
        assert executor.position.quantity == 0  # ポジションが決済されているはず

    def test_update_equity(self, executor):
        """資産価値更新のテスト"""
        current_price = 100
        timestamp = datetime.now()

        executor.update_equity(current_price, timestamp)

        assert len(executor.equity_curve) == 1
        assert executor.equity_curve[0]['equity'] == executor.initial_capital
        assert executor.equity_curve[0]['timestamp'] == timestamp

    def test_evaluate_simple_condition(self, executor, sample_data):
        """簡単な条件評価のテスト"""
        # 指標を計算
        indicators_config = [{'name': 'SMA', 'params': {'period': 10}}]
        executor.calculate_indicators(sample_data, indicators_config)

        # 現在価格 > 100 の条件をテスト
        condition = "close > 100"
        result = executor.evaluate_condition(condition, 10, sample_data)

        # sample_dataの10番目の終値が100より大きいかどうかで判定
        expected = sample_data.iloc[10]['close'] > 100
        assert result == expected

    def test_simple_strategy_backtest(self, executor, sample_data):
        """シンプルな戦略のバックテストテスト"""
        strategy_config = {
            'indicators': [
                {'name': 'SMA', 'params': {'period': 5}},
                {'name': 'SMA', 'params': {'period': 10}}
            ],
            'entry_rules': [
                {'condition': 'close > 100'}  # 簡単な条件
            ],
            'exit_rules': [
                {'condition': 'close > 115'}  # 利確条件
            ]
        }

        result = executor.run_backtest(sample_data, strategy_config)

        # 結果の基本的な検証
        assert 'total_return' in result
        assert 'sharpe_ratio' in result
        assert 'max_drawdown' in result
        assert 'win_rate' in result
        assert 'total_trades' in result
        assert 'equity_curve' in result
        assert 'trades' in result

        # 資産価値の記録があることを確認
        assert len(result['equity_curve']) == len(sample_data)

    def test_performance_metrics_calculation(self, executor):
        """パフォーマンス指標計算のテスト"""
        # ダミーの資産価値曲線を作成
        for i in range(10):
            executor.equity_curve.append({
                'equity': 100000 + i * 1000,  # 毎日1000円増加
                'timestamp': datetime.now()
            })

        # ダミーの取引履歴を作成
        executor.trades = [
            Trade(datetime.now(), 'buy', 100, 100, 10),
            Trade(datetime.now(), 'sell', 110, 100, 10, pnl=1000),
            Trade(datetime.now(), 'buy', 105, 100, 10),
            Trade(datetime.now(), 'sell', 95, 100, 10, pnl=-1000),
        ]

        metrics = executor.calculate_performance_metrics()

        assert 'total_return' in metrics
        assert 'win_rate' in metrics
        assert metrics['total_trades'] == 2  # 売り取引の数
        assert metrics['winning_trades'] == 1
        assert metrics['losing_trades'] == 1
        assert metrics['win_rate'] == 0.5


if __name__ == "__main__":
    pytest.main([__file__])
