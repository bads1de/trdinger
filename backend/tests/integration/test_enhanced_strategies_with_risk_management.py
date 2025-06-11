"""
リスク管理機能付き戦略の統合テスト

実際のバックテストでリスク管理機能が正常に動作することを確認
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backtesting import Backtest
from app.core.strategies.enhanced_sma_cross_strategy import (
    EnhancedSMACrossStrategy,
    EnhancedSMACrossStrategyWithTrailing,
    EnhancedSMACrossStrategyWithVolume,
    EnhancedSMACrossStrategyAdvanced,
)
from app.core.strategies.sma_cross_strategy import SMACrossStrategy


class TestEnhancedStrategiesWithRiskManagement:
    """リスク管理機能付き戦略の統合テスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータ"""
        dates = pd.date_range("2023-01-01", periods=200, freq="D")
        np.random.seed(42)

        # より複雑なトレンドデータを生成
        base_price = 100
        trend = np.linspace(0, 30, 200)  # 上昇トレンド
        noise = np.random.normal(0, 3, 200)
        cycle = 10 * np.sin(np.linspace(0, 4 * np.pi, 200))  # サイクル成分
        prices = base_price + trend + noise + cycle

        # OHLCV データを生成
        data = pd.DataFrame(
            {
                "Open": prices * (1 + np.random.normal(0, 0.01, 200)),
                "High": prices * (1 + np.abs(np.random.normal(0, 0.02, 200))),
                "Low": prices * (1 - np.abs(np.random.normal(0, 0.02, 200))),
                "Close": prices,
                "Volume": np.random.randint(1000, 10000, 200).astype(
                    float
                ),  # float型に変換
            },
            index=dates,
        )

        return data

    def test_enhanced_sma_cross_strategy_basic(self, sample_data):
        """基本的なリスク管理機能付きSMAクロス戦略のテスト"""

        bt = Backtest(
            sample_data, EnhancedSMACrossStrategy, cash=10000, commission=0.001
        )

        stats = bt.run(n1=10, n2=30, sl_pct=0.02, tp_pct=0.05, use_risk_management=True)

        # 基本的な結果の検証
        assert stats is not None
        assert "Equity Final [$]" in stats
        assert "# Trades" in stats
        assert stats["# Trades"] > 0

        # リスク管理が適用されていることを確認
        # （取引数が従来戦略より多い可能性がある）
        print(f"Enhanced strategy trades: {stats['# Trades']}")
        print(f"Final equity: ${stats['Equity Final [$]']:.2f}")
        print(f"Max drawdown: {stats['Max. Drawdown [%]']:.2f}%")

    def test_enhanced_vs_basic_strategy_comparison(self, sample_data):
        """リスク管理機能付き戦略と基本戦略の比較テスト"""

        # 基本戦略
        bt_basic = Backtest(sample_data, SMACrossStrategy, cash=10000, commission=0.001)
        stats_basic = bt_basic.run(n1=10, n2=30)

        # リスク管理機能付き戦略
        bt_enhanced = Backtest(
            sample_data, EnhancedSMACrossStrategy, cash=10000, commission=0.001
        )
        stats_enhanced = bt_enhanced.run(
            n1=10, n2=30, sl_pct=0.02, tp_pct=0.05, use_risk_management=True
        )

        # 両方の戦略が正常に実行されることを確認
        assert stats_basic is not None
        assert stats_enhanced is not None

        # 結果の比較
        print("\n=== Strategy Comparison ===")
        print(f"Basic Strategy:")
        print(f"  Trades: {stats_basic['# Trades']}")
        print(f"  Final Equity: ${stats_basic['Equity Final [$]']:.2f}")
        print(f"  Max Drawdown: {stats_basic['Max. Drawdown [%]']:.2f}%")

        print(f"Enhanced Strategy:")
        print(f"  Trades: {stats_enhanced['# Trades']}")
        print(f"  Final Equity: ${stats_enhanced['Equity Final [$]']:.2f}")
        print(f"  Max Drawdown: {stats_enhanced['Max. Drawdown [%]']:.2f}%")

        # リスク管理により取引数が変わる可能性がある
        assert stats_enhanced["# Trades"] >= 0

    def test_trailing_stop_strategy(self, sample_data):
        """トレーリングストップ機能付き戦略のテスト"""

        bt = Backtest(
            sample_data,
            EnhancedSMACrossStrategyWithTrailing,
            cash=10000,
            commission=0.001,
        )

        stats = bt.run(
            n1=10,
            n2=30,
            sl_pct=0.02,
            tp_pct=0.05,
            use_risk_management=True,
            use_trailing_stop=True,
            trailing_update_frequency=1,
        )

        # 基本的な結果の検証
        assert stats is not None
        assert "Equity Final [$]" in stats
        assert "# Trades" in stats

        print(f"\nTrailing Stop Strategy:")
        print(f"  Trades: {stats['# Trades']}")
        print(f"  Final Equity: ${stats['Equity Final [$]']:.2f}")
        print(f"  Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")

    def test_volume_filter_strategy(self, sample_data):
        """出来高フィルター付き戦略のテスト"""

        bt = Backtest(
            sample_data,
            EnhancedSMACrossStrategyWithVolume,
            cash=10000,
            commission=0.001,
        )

        stats = bt.run(
            n1=10,
            n2=30,
            sl_pct=0.02,
            tp_pct=0.05,
            use_risk_management=True,
            volume_threshold=1.2,
            volume_period=20,
        )

        # 基本的な結果の検証
        assert stats is not None
        assert "Equity Final [$]" in stats
        assert "# Trades" in stats

        print(f"\nVolume Filter Strategy:")
        print(f"  Trades: {stats['# Trades']}")
        print(f"  Final Equity: ${stats['Equity Final [$]']:.2f}")
        print(f"  Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")

    def test_advanced_strategy(self, sample_data):
        """高度な戦略のテスト"""

        bt = Backtest(
            sample_data, EnhancedSMACrossStrategyAdvanced, cash=10000, commission=0.001
        )

        stats = bt.run(
            n1=10,
            n2=30,
            sl_pct=0.02,
            tp_pct=0.05,
            use_risk_management=True,
            use_trailing_stop=True,
            use_volume_filter=True,
            volume_threshold=1.2,
            trailing_update_frequency=1,
            use_atr_based_risk=False,  # パーセンテージベースを使用
        )

        # 基本的な結果の検証
        assert stats is not None
        assert "Equity Final [$]" in stats
        assert "# Trades" in stats

        print(f"\nAdvanced Strategy:")
        print(f"  Trades: {stats['# Trades']}")
        print(f"  Final Equity: ${stats['Equity Final [$]']:.2f}")
        print(f"  Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")

    def test_risk_management_disabled(self, sample_data):
        """リスク管理機能を無効にした場合のテスト"""

        bt = Backtest(
            sample_data, EnhancedSMACrossStrategy, cash=10000, commission=0.001
        )

        stats = bt.run(n1=10, n2=30, use_risk_management=False)

        # 基本的な結果の検証
        assert stats is not None
        assert "Equity Final [$]" in stats
        assert "# Trades" in stats

        print(f"\nRisk Management Disabled:")
        print(f"  Trades: {stats['# Trades']}")
        print(f"  Final Equity: ${stats['Equity Final [$]']:.2f}")
        print(f"  Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")

    def test_parameter_validation(self, sample_data):
        """パラメータ検証のテスト"""

        # 無効なパラメータでのバックテストテスト
        bt = Backtest(
            sample_data, EnhancedSMACrossStrategy, cash=10000, commission=0.001
        )

        # 無効なSMA期間でのテスト
        with pytest.raises(ValueError):
            bt.run(n1=50, n2=20, use_risk_management=True)  # n1 > n2 は無効

        # 無効なリスク管理パラメータでのテスト
        with pytest.raises(ValueError):
            bt.run(
                n1=10,
                n2=30,
                sl_pct=1.5,  # 100%を超える値は無効
                use_risk_management=True,
            )

    def test_multiple_strategies_performance(self, sample_data):
        """複数戦略のパフォーマンス比較テスト"""

        strategies = {
            "Basic": SMACrossStrategy,
            "Enhanced": EnhancedSMACrossStrategy,
            "WithTrailing": EnhancedSMACrossStrategyWithTrailing,
            "WithVolume": EnhancedSMACrossStrategyWithVolume,
            "Advanced": EnhancedSMACrossStrategyAdvanced,
        }

        results = {}

        for name, strategy_class in strategies.items():
            bt = Backtest(sample_data, strategy_class, cash=10000, commission=0.001)

            if name == "Basic":
                stats = bt.run(n1=10, n2=30)
            else:
                stats = bt.run(
                    n1=10, n2=30, sl_pct=0.02, tp_pct=0.05, use_risk_management=True
                )

            results[name] = {
                "trades": stats["# Trades"],
                "final_equity": stats["Equity Final [$]"],
                "max_drawdown": stats["Max. Drawdown [%]"],
                "return_pct": stats["Return [%]"],
            }

        # 結果の表示
        print("\n=== Multi-Strategy Performance Comparison ===")
        for name, metrics in results.items():
            print(f"{name}:")
            print(f"  Trades: {metrics['trades']}")
            print(f"  Final Equity: ${metrics['final_equity']:.2f}")
            print(f"  Return: {metrics['return_pct']:.2f}%")
            print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
            print()

        # 全ての戦略が正常に実行されたことを確認
        for name, metrics in results.items():
            assert metrics["trades"] >= 0
            assert metrics["final_equity"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
