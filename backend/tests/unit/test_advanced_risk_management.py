"""
高度なリスク管理機能のテスト

Kelly Criterion、Risk-Reward Ratio、Position Sizingなどの
高度なリスク管理機能をテスト
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
from app.core.strategies.advanced_risk_management_strategy import (
    AdvancedRiskManagementStrategy,
    ConservativeRiskManagementStrategy,
    AggressiveRiskManagementStrategy
)
from app.core.strategies.risk_management import RiskCalculator


class TestAdvancedRiskManagement:
    """高度なリスク管理機能のテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータ"""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        np.random.seed(42)
        
        # より複雑なトレンドデータを生成
        base_price = 100
        trend = np.linspace(0, 40, 300)  # 上昇トレンド
        noise = np.random.normal(0, 2, 300)
        cycle = 15 * np.sin(np.linspace(0, 6*np.pi, 300))  # サイクル成分
        volatility = 5 * np.random.random(300)  # ランダムボラティリティ
        
        prices = base_price + trend + noise + cycle + volatility
        
        # OHLCV データを生成
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.01, 300)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.025, 300))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.025, 300))),
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, 300).astype(float)
        }, index=dates)
        
        return data

    def test_kelly_criterion_calculation(self):
        """Kelly Criterion計算のテスト"""
        
        calculator = RiskCalculator()
        
        # 基本的なKelly計算
        kelly_ratio = calculator.calculate_kelly_criterion(
            win_rate=0.6,  # 60%勝率
            avg_win=0.05,  # 平均5%利益
            avg_loss=0.03  # 平均3%損失
        )
        
        assert kelly_ratio is not None
        assert 0 <= kelly_ratio <= 1
        print(f"Kelly ratio (60% win, 5% avg win, 3% avg loss): {kelly_ratio:.3f}")
        
        # 勝率が低い場合
        kelly_ratio_low = calculator.calculate_kelly_criterion(
            win_rate=0.3,  # 30%勝率
            avg_win=0.10,  # 平均10%利益
            avg_loss=0.02  # 平均2%損失
        )
        
        assert kelly_ratio_low is not None
        print(f"Kelly ratio (30% win, 10% avg win, 2% avg loss): {kelly_ratio_low:.3f}")

    def test_risk_reward_ratio_calculation(self):
        """リスクリワード比率計算のテスト"""
        
        calculator = RiskCalculator()
        
        # ロングポジションのリスクリワード比率
        rr_ratio = calculator.calculate_risk_reward_ratio(
            entry_price=100.0,
            sl_price=98.0,    # 2%リスク
            tp_price=106.0,   # 6%リワード
            is_long=True
        )
        
        assert rr_ratio is not None
        assert rr_ratio == 3.0  # 6% / 2% = 3:1
        print(f"Risk-Reward ratio (Long): {rr_ratio:.1f}:1")
        
        # ショートポジションのリスクリワード比率
        rr_ratio_short = calculator.calculate_risk_reward_ratio(
            entry_price=100.0,
            sl_price=102.0,   # 2%リスク
            tp_price=95.0,    # 5%リワード
            is_long=False
        )
        
        assert rr_ratio_short is not None
        assert rr_ratio_short == 2.5  # 5% / 2% = 2.5:1
        print(f"Risk-Reward ratio (Short): {rr_ratio_short:.1f}:1")

    def test_position_sizing_methods(self):
        """ポジションサイジング方法のテスト"""
        
        calculator = RiskCalculator()
        current_equity = 10000
        entry_price = 100.0
        sl_price = 98.0
        
        # 固定比率方式
        size_fixed_ratio = calculator.calculate_optimal_position_size(
            current_equity, entry_price, sl_price,
            method="fixed_ratio", ratio=0.02
        )
        assert size_fixed_ratio is not None
        print(f"Position size (fixed ratio 2%): {size_fixed_ratio:.2f}")
        
        # 固定リスク方式
        size_fixed_risk = calculator.calculate_optimal_position_size(
            current_equity, entry_price, sl_price,
            method="fixed_risk", risk_amount=100  # $100リスク
        )
        assert size_fixed_risk is not None
        print(f"Position size (fixed risk $100): {size_fixed_risk:.2f}")
        
        # Kelly方式
        size_kelly = calculator.calculate_optimal_position_size(
            current_equity, entry_price, sl_price,
            method="kelly", win_rate=0.6, avg_win=0.05, avg_loss=0.03
        )
        assert size_kelly is not None
        print(f"Position size (Kelly): {size_kelly:.2f}")

    def test_advanced_strategy_basic_functionality(self, sample_data):
        """高度なリスク管理戦略の基本機能テスト"""
        
        bt = Backtest(
            sample_data, 
            AdvancedRiskManagementStrategy, 
            cash=10000, 
            commission=0.001
        )
        
        stats = bt.run(
            n1=10,
            n2=30,
            sl_pct=0.02,
            tp_pct=0.05,
            min_risk_reward_ratio=2.0,
            use_kelly_criterion=True,
            position_sizing_method="kelly"
        )
        
        # 基本的な結果の検証
        assert stats is not None
        assert 'Equity Final [$]' in stats
        assert '# Trades' in stats
        
        print(f"\nAdvanced Strategy Results:")
        print(f"  Trades: {stats['# Trades']}")
        print(f"  Final Equity: ${stats['Equity Final [$]']:.2f}")
        print(f"  Return: {stats['Return [%]']:.2f}%")
        print(f"  Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")

    def test_conservative_vs_aggressive_strategies(self, sample_data):
        """保守的戦略 vs アグレッシブ戦略の比較テスト"""
        
        # 保守的戦略
        bt_conservative = Backtest(
            sample_data, 
            ConservativeRiskManagementStrategy, 
            cash=10000, 
            commission=0.001
        )
        stats_conservative = bt_conservative.run(n1=10, n2=30)
        
        # アグレッシブ戦略
        bt_aggressive = Backtest(
            sample_data, 
            AggressiveRiskManagementStrategy, 
            cash=10000, 
            commission=0.001
        )
        stats_aggressive = bt_aggressive.run(n1=10, n2=30)
        
        # 結果の比較
        print(f"\n=== Strategy Comparison ===")
        print(f"Conservative Strategy:")
        print(f"  Trades: {stats_conservative['# Trades']}")
        print(f"  Final Equity: ${stats_conservative['Equity Final [$]']:.2f}")
        print(f"  Return: {stats_conservative['Return [%]']:.2f}%")
        print(f"  Max Drawdown: {stats_conservative['Max. Drawdown [%]']:.2f}%")
        
        print(f"Aggressive Strategy:")
        print(f"  Trades: {stats_aggressive['# Trades']}")
        print(f"  Final Equity: ${stats_aggressive['Equity Final [$]']:.2f}")
        print(f"  Return: {stats_aggressive['Return [%]']:.2f}%")
        print(f"  Max Drawdown: {stats_aggressive['Max. Drawdown [%]']:.2f}%")
        
        # 両方の戦略が正常に実行されることを確認
        assert stats_conservative is not None
        assert stats_aggressive is not None

    def test_risk_reward_filtering(self, sample_data):
        """リスクリワード比率フィルタリングのテスト"""
        
        # 高いリスクリワード比率要求（厳しいフィルタリング）
        bt_strict = Backtest(
            sample_data, 
            AdvancedRiskManagementStrategy, 
            cash=10000, 
            commission=0.001
        )
        stats_strict = bt_strict.run(
            n1=10, n2=30,
            min_risk_reward_ratio=3.0,  # 3:1以上要求
            use_kelly_criterion=False
        )
        
        # 低いリスクリワード比率要求（緩いフィルタリング）
        bt_loose = Backtest(
            sample_data, 
            AdvancedRiskManagementStrategy, 
            cash=10000, 
            commission=0.001
        )
        stats_loose = bt_loose.run(
            n1=10, n2=30,
            min_risk_reward_ratio=1.0,  # 1:1以上要求
            use_kelly_criterion=False
        )
        
        print(f"\n=== Risk-Reward Filtering Comparison ===")
        print(f"Strict Filtering (3:1 min):")
        print(f"  Trades: {stats_strict['# Trades']}")
        print(f"  Final Equity: ${stats_strict['Equity Final [$]']:.2f}")
        
        print(f"Loose Filtering (1:1 min):")
        print(f"  Trades: {stats_loose['# Trades']}")
        print(f"  Final Equity: ${stats_loose['Equity Final [$]']:.2f}")
        
        # 厳しいフィルタリングの方が取引数が少ないはず
        # assert stats_strict['# Trades'] <= stats_loose['# Trades']

    def test_atr_based_risk_management(self, sample_data):
        """ATRベースリスク管理のテスト"""
        
        bt = Backtest(
            sample_data, 
            AdvancedRiskManagementStrategy, 
            cash=10000, 
            commission=0.001
        )
        
        stats = bt.run(
            n1=10, n2=30,
            use_atr_based_risk=True,
            atr_sl_multiplier=2.0,
            atr_tp_multiplier=3.0,
            min_risk_reward_ratio=1.5
        )
        
        print(f"\n=== ATR-Based Risk Management ===")
        print(f"  Trades: {stats['# Trades']}")
        print(f"  Final Equity: ${stats['Equity Final [$]']:.2f}")
        print(f"  Return: {stats['Return [%]']:.2f}%")
        print(f"  Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
        
        assert stats is not None
        assert 'Equity Final [$]' in stats

    def test_position_sizing_comparison(self, sample_data):
        """異なるポジションサイジング方法の比較テスト"""
        
        methods = ["fixed_ratio", "fixed_risk", "kelly"]
        results = {}
        
        for method in methods:
            bt = Backtest(
                sample_data, 
                AdvancedRiskManagementStrategy, 
                cash=10000, 
                commission=0.001
            )
            
            stats = bt.run(
                n1=10, n2=30,
                position_sizing_method=method,
                use_kelly_criterion=(method == "kelly")
            )
            
            results[method] = {
                'trades': stats['# Trades'],
                'final_equity': stats['Equity Final [$]'],
                'return_pct': stats['Return [%]'],
                'max_drawdown': stats['Max. Drawdown [%]']
            }
        
        print(f"\n=== Position Sizing Method Comparison ===")
        for method, metrics in results.items():
            print(f"{method.title()}:")
            print(f"  Trades: {metrics['trades']}")
            print(f"  Final Equity: ${metrics['final_equity']:.2f}")
            print(f"  Return: {metrics['return_pct']:.2f}%")
            print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
            print()
        
        # 全ての方法が正常に実行されることを確認
        for method in methods:
            assert results[method]['trades'] >= 0
            assert results[method]['final_equity'] > 0

    def test_invalid_parameters_handling(self):
        """無効なパラメータの処理テスト"""
        
        calculator = RiskCalculator()
        
        # 無効なKelly Criterion パラメータ
        kelly_invalid = calculator.calculate_kelly_criterion(
            win_rate=1.5,  # 無効（1を超える）
            avg_win=0.05,
            avg_loss=0.03
        )
        assert kelly_invalid is None
        
        # 無効なリスクリワード比率パラメータ
        rr_invalid = calculator.calculate_risk_reward_ratio(
            entry_price=100.0,
            sl_price=105.0,  # 無効（ロングでSLがエントリーより高い）
            tp_price=95.0,   # 無効（ロングでTPがエントリーより低い）
            is_long=True
        )
        assert rr_invalid is None
        
        # 無効なポジションサイズパラメータ
        size_invalid = calculator.calculate_optimal_position_size(
            current_equity=-1000,  # 無効（負の値）
            entry_price=100.0,
            sl_price=98.0,
            method="fixed_ratio"
        )
        assert size_invalid is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
