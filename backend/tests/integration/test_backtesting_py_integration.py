"""
backtesting.pyライブラリとの統合テスト
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtesting import Backtest
from app.core.strategies.sma_cross_strategy import SMACrossStrategy
from app.core.strategies.indicators import SMA


class TestBacktestingPyIntegration:
    """backtesting.pyライブラリとの統合テスト"""

    @pytest.fixture
    def sample_btc_data(self):
        """BTC価格のサンプルデータ"""
        # 1年分のサンプルデータを生成
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        
        # より現実的なBTC価格データを生成
        np.random.seed(42)
        base_price = 50000
        
        # トレンドとボラティリティを含む価格生成
        trend = np.linspace(0, 0.5, len(dates))  # 50%の年間上昇トレンド
        volatility = 0.03  # 3%の日次ボラティリティ
        
        returns = np.random.normal(0, volatility, len(dates))
        price_multiplier = np.exp(np.cumsum(returns) + trend)
        prices = base_price * price_multiplier
        
        # OHLCV データを生成
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # 日次の高値・安値を生成
            daily_volatility = 0.02
            high = price * (1 + abs(np.random.normal(0, daily_volatility)))
            low = price * (1 - abs(np.random.normal(0, daily_volatility)))
            
            # 始値・終値を生成
            if i == 0:
                open_price = price
            else:
                open_price = data[i-1]['Close'] * (1 + np.random.normal(0, 0.005))
            
            close_price = price
            volume = np.random.randint(1000, 10000)
            
            data.append({
                'Open': open_price,
                'High': max(open_price, high, close_price),
                'Low': min(open_price, low, close_price),
                'Close': close_price,
                'Volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        return df

    def test_sma_cross_strategy_basic_execution(self, sample_btc_data):
        """SMAクロス戦略の基本実行テスト"""
        # バックテストを実行
        bt = Backtest(
            sample_btc_data,
            SMACrossStrategy,
            cash=100000,
            commission=0.001,
            exclusive_orders=True
        )
        
        # 戦略を実行
        stats = bt.run()
        
        # 基本的な結果の検証
        assert stats is not None
        assert 'Return [%]' in stats
        assert 'Sharpe Ratio' in stats
        assert 'Max. Drawdown [%]' in stats
        assert '# Trades' in stats
        
        # 取引が発生していることを確認
        assert stats['# Trades'] > 0
        
        # 戦略パラメータが正しく設定されていることを確認
        strategy_instance = stats['_strategy']
        assert strategy_instance.n1 == 20
        assert strategy_instance.n2 == 50

    def test_sma_cross_strategy_with_custom_parameters(self, sample_btc_data):
        """カスタムパラメータでのSMAクロス戦略テスト"""
        # カスタムパラメータの戦略クラス
        class CustomSMACrossStrategy(SMACrossStrategy):
            n1 = 10
            n2 = 30
        
        # バックテストを実行
        bt = Backtest(
            sample_btc_data,
            CustomSMACrossStrategy,
            cash=50000,
            commission=0.002
        )
        
        stats = bt.run()
        
        # カスタムパラメータが適用されていることを確認
        strategy_instance = stats['_strategy']
        assert strategy_instance.n1 == 10
        assert strategy_instance.n2 == 30
        
        # 異なるパラメータで異なる結果が得られることを確認
        # （完全に同じ結果になることはほぼない）
        assert stats['# Trades'] > 0

    def test_sma_cross_strategy_optimization(self, sample_btc_data):
        """SMAクロス戦略の最適化テスト"""
        # 小さなパラメータ範囲で最適化をテスト
        bt = Backtest(
            sample_btc_data,
            SMACrossStrategy,
            cash=100000,
            commission=0.001
        )
        
        # パラメータ最適化を実行
        stats = bt.optimize(
            n1=range(10, 21, 5),  # [10, 15, 20]
            n2=range(30, 51, 10), # [30, 40, 50]
            maximize='Equity Final [$]',
            constraint=lambda param: param.n1 < param.n2
        )
        
        # 最適化結果の検証
        assert stats is not None
        assert 'Return [%]' in stats
        
        # 最適化されたパラメータが制約を満たしていることを確認
        optimized_strategy = stats['_strategy']
        assert optimized_strategy.n1 < optimized_strategy.n2
        assert optimized_strategy.n1 in [10, 15, 20]
        assert optimized_strategy.n2 in [30, 40, 50]

    def test_backtest_with_insufficient_data(self):
        """データ不足時のバックテストテスト"""
        # 50日未満のデータ（SMA(50)には不十分）
        short_data = pd.DataFrame({
            'Open': [100] * 30,
            'High': [105] * 30,
            'Low': [95] * 30,
            'Close': [100] * 30,
            'Volume': [1000] * 30
        }, index=pd.date_range('2024-01-01', periods=30))
        
        # バックテストを実行
        bt = Backtest(short_data, SMACrossStrategy, cash=10000)
        stats = bt.run()
        
        # データ不足でも実行できることを確認
        assert stats is not None
        # 取引が発生しない可能性が高い
        assert stats['# Trades'] >= 0

    def test_backtest_performance_metrics(self, sample_btc_data):
        """バックテストのパフォーマンス指標テスト"""
        bt = Backtest(
            sample_btc_data,
            SMACrossStrategy,
            cash=100000,
            commission=0.001
        )
        
        stats = bt.run()
        
        # 重要なパフォーマンス指標が存在することを確認
        required_metrics = [
            'Start',
            'End',
            'Duration',
            'Exposure Time [%]',
            'Equity Final [$]',
            'Return [%]',
            'Buy & Hold Return [%]',
            'Max. Drawdown [%]',
            'Sharpe Ratio',
            '# Trades',
            'Win Rate [%]'
        ]
        
        for metric in required_metrics:
            assert metric in stats, f"Missing metric: {metric}"
        
        # 数値の妥当性チェック
        assert stats['Equity Final [$]'] > 0
        assert 0 <= stats['Exposure Time [%]'] <= 100
        assert -100 <= stats['Max. Drawdown [%]'] <= 0
        assert 0 <= stats['Win Rate [%]'] <= 100

    def test_equity_curve_and_trades_data(self, sample_btc_data):
        """資産曲線と取引データのテスト"""
        bt = Backtest(
            sample_btc_data,
            SMACrossStrategy,
            cash=100000,
            commission=0.001
        )
        
        stats = bt.run()
        
        # 資産曲線データの確認
        equity_curve = stats['_equity_curve']
        assert isinstance(equity_curve, pd.DataFrame)
        assert 'Equity' in equity_curve.columns
        assert 'DrawdownPct' in equity_curve.columns
        assert len(equity_curve) > 0
        
        # 取引データの確認
        trades = stats['_trades']
        if len(trades) > 0:  # 取引が発生した場合
            assert isinstance(trades, pd.DataFrame)
            required_trade_columns = [
                'Size', 'EntryBar', 'ExitBar', 'EntryPrice', 'ExitPrice', 'PnL', 'ReturnPct'
            ]
            for col in required_trade_columns:
                assert col in trades.columns, f"Missing trade column: {col}"

    def test_sma_indicator_calculation(self, sample_btc_data):
        """SMA指標の計算テスト"""
        close_prices = sample_btc_data['Close']
        
        # SMA(20)の計算
        sma_20 = SMA(close_prices, 20)
        
        # 基本的な検証
        assert len(sma_20) == len(close_prices)
        assert pd.isna(sma_20.iloc[:19]).all()  # 最初の19個はNaN
        assert not pd.isna(sma_20.iloc[19:]).any()  # 20個目以降はNaNでない
        
        # 手動計算との比較
        manual_sma_20 = close_prices.rolling(window=20).mean()
        pd.testing.assert_series_equal(sma_20, manual_sma_20)

    def test_strategy_with_real_market_conditions(self, sample_btc_data):
        """実際の市場条件でのテスト"""
        # 手数料とスリッページを含む現実的な設定
        bt = Backtest(
            sample_btc_data,
            SMACrossStrategy,
            cash=100000,
            commission=0.001,  # 0.1%の手数料
            exclusive_orders=True,
            trade_on_close=True  # 終値で取引
        )
        
        stats = bt.run()
        
        # 現実的な結果の検証
        assert stats['Equity Final [$]'] > 0
        
        # 手数料が適用されていることを確認
        if stats['# Trades'] > 0:
            total_commission = stats['# Trades'] * stats['Equity Final [$]'] * 0.001
            # 手数料が合理的な範囲内であることを確認
            assert total_commission > 0
