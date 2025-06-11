"""
backtesting.pyのSL/TP機能を活用したリスク管理統合テスト

TDD（テスト駆動開発）アプローチ：
1. 失敗するテストケースを作成
2. テストが通る最小限のコードを実装
3. リファクタリングで品質向上
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from app.core.strategies.indicators import SMA, RSI


class TestRiskManagementIntegration:
    """backtesting.pyのSL/TP機能統合テスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータ"""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        np.random.seed(42)

        # トレンドのあるデータを生成
        base_price = 100
        trend = np.linspace(0, 20, 100)  # 上昇トレンド
        noise = np.random.normal(0, 2, 100)
        prices = base_price + trend + noise

        # OHLCV データを生成
        data = pd.DataFrame(
            {
                "Open": prices * (1 + np.random.normal(0, 0.01, 100)),
                "High": prices * (1 + np.abs(np.random.normal(0, 0.02, 100))),
                "Low": prices * (1 - np.abs(np.random.normal(0, 0.02, 100))),
                "Close": prices,
                "Volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )

        return data

    def test_buy_with_sl_tp_parameters(self, sample_data):
        """買い注文でSL/TPパラメータが正しく設定されるかテスト"""

        class TestStrategy(Strategy):
            def init(self):
                self.sma = self.I(SMA, self.data.Close, 10)

            def next(self):
                if len(self.sma) < 10:
                    return

                current_price = self.data.Close[-1]

                # SL/TPを指定した買い注文
                if not self.position and current_price > self.sma[-1]:
                    # 2%のストップロス、5%のテイクプロフィット
                    sl_price = current_price * 0.98
                    tp_price = current_price * 1.05
                    self.buy(sl=sl_price, tp=tp_price)

        bt = Backtest(sample_data, TestStrategy, cash=10000, commission=0.001)

        # バックテストが正常に実行されることを確認
        stats = bt.run()

        # 基本的な結果の検証
        assert stats is not None
        assert "Equity Final [$]" in stats
        assert "# Trades" in stats
        assert stats["# Trades"] > 0  # 取引が発生していることを確認

    def test_sell_with_sl_tp_parameters(self, sample_data):
        """売り注文でSL/TPパラメータが正しく設定されるかテスト"""

        class TestStrategy(Strategy):
            def init(self):
                self.sma = self.I(SMA, self.data.Close, 10)

            def next(self):
                if len(self.sma) < 10:
                    return

                current_price = self.data.Close[-1]

                # SL/TPを指定した売り注文
                if not self.position and current_price < self.sma[-1]:
                    # 2%のストップロス、5%のテイクプロフィット
                    sl_price = current_price * 1.02
                    tp_price = current_price * 0.95
                    self.sell(sl=sl_price, tp=tp_price)

        bt = Backtest(sample_data, TestStrategy, cash=10000, commission=0.001)

        # バックテストが正常に実行されることを確認
        stats = bt.run()

        # 基本的な結果の検証
        assert stats is not None
        assert "Equity Final [$]" in stats

    def test_dynamic_sl_tp_modification(self, sample_data):
        """動的なSL/TP変更のテスト"""

        class TestStrategy(Strategy):
            def init(self):
                self.sma = self.I(SMA, self.data.Close, 10)

            def next(self):
                if len(self.sma) < 10:
                    return

                current_price = self.data.Close[-1]

                # エントリー
                if not self.position and current_price > self.sma[-1]:
                    self.buy()

                # 動的SL/TP調整
                if self.position:
                    for trade in self.trades:
                        # トレーリングストップロスの実装
                        new_sl = current_price * 0.98
                        trade.sl = max(trade.sl or 0, new_sl)

        bt = Backtest(sample_data, TestStrategy, cash=10000, commission=0.001)

        # バックテストが正常に実行されることを確認
        stats = bt.run()

        # 基本的な結果の検証
        assert stats is not None
        assert "Equity Final [$]" in stats

    def test_percentage_based_sl_tp_calculation(self):
        """パーセンテージベースのSL/TP計算テスト"""

        from app.core.strategies.risk_management import calculate_sl_tp_prices

        entry_price = 100.0
        sl_pct = 0.02  # 2%
        tp_pct = 0.05  # 5%

        sl_price, tp_price = calculate_sl_tp_prices(
            entry_price, sl_pct, tp_pct, is_long=True
        )

        assert sl_price == 98.0
        assert tp_price == 105.0

    def test_absolute_price_sl_tp_calculation(self):
        """絶対価格ベースのSL/TP計算テスト"""

        from app.core.strategies.risk_management import calculate_sl_tp_prices

        entry_price = 100.0
        sl_price = 95.0
        tp_price = 110.0

        result_sl, result_tp = calculate_sl_tp_prices(
            entry_price, sl_price, tp_price, is_long=True, use_absolute=True
        )

        assert result_sl == 95.0
        assert result_tp == 110.0

    def test_risk_management_mixin_integration(self, sample_data):
        """リスク管理Mixinの統合テスト"""

        from app.core.strategies.risk_management import RiskManagementMixin

        class TestStrategy(Strategy, RiskManagementMixin):
            def init(self):
                self.sma = self.I(SMA, self.data.Close, 10)
                self.setup_risk_management(sl_pct=0.02, tp_pct=0.05)

            def next(self):
                if len(self.sma) < 10:
                    return

                current_price = self.data.Close[-1]

                if not self.position and current_price > self.sma[-1]:
                    self.buy_with_risk_management()

        bt = Backtest(sample_data, TestStrategy, cash=10000, commission=0.001)
        stats = bt.run()

        # 基本的な結果の検証
        assert stats is not None
        assert "Equity Final [$]" in stats
        assert "# Trades" in stats

    def test_backtest_results_with_risk_management(self, sample_data):
        """リスク管理適用時のバックテスト結果テスト（失敗予定）"""

        # 基本戦略（リスク管理なし）
        class BasicStrategy(Strategy):
            def init(self):
                self.sma = self.I(SMA, self.data.Close, 10)

            def next(self):
                if len(self.sma) < 10:
                    return

                current_price = self.data.Close[-1]

                if not self.position and current_price > self.sma[-1]:
                    self.buy()
                elif self.position and current_price < self.sma[-1]:
                    self.position.close()

        # リスク管理戦略
        class RiskManagedStrategy(Strategy):
            def init(self):
                self.sma = self.I(SMA, self.data.Close, 10)

            def next(self):
                if len(self.sma) < 10:
                    return

                current_price = self.data.Close[-1]

                if not self.position and current_price > self.sma[-1]:
                    # SL/TP付きの注文
                    sl_price = current_price * 0.98
                    tp_price = current_price * 1.05
                    self.buy(sl=sl_price, tp=tp_price)

        # 基本戦略のテスト
        bt_basic = Backtest(sample_data, BasicStrategy, cash=10000, commission=0.001)
        stats_basic = bt_basic.run()

        # リスク管理戦略のテスト
        bt_risk = Backtest(
            sample_data, RiskManagedStrategy, cash=10000, commission=0.001
        )
        stats_risk = bt_risk.run()

        # 両方の戦略が正常に実行されることを確認
        assert stats_basic is not None
        assert stats_risk is not None
        assert "Equity Final [$]" in stats_basic
        assert "Equity Final [$]" in stats_risk
        assert "# Trades" in stats_basic
        assert "# Trades" in stats_risk


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
