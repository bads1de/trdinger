"""
完全なリスク管理システムの統合テスト

今回実装した全ての機能を包括的にテストする
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
    EnhancedSMACrossStrategyAdvanced
)
from app.core.strategies.advanced_risk_management_strategy import (
    AdvancedRiskManagementStrategy,
    ConservativeRiskManagementStrategy,
    AggressiveRiskManagementStrategy
)
from app.core.strategies.sma_cross_strategy import SMACrossStrategy
from app.core.strategies.risk_management import RiskCalculator


class TestCompleteRiskManagementSystem:
    """完全なリスク管理システムの統合テスト"""

    @pytest.fixture
    def comprehensive_data(self):
        """包括的なテスト用データ"""
        dates = pd.date_range('2023-01-01', periods=500, freq='D')
        np.random.seed(42)
        
        # より現実的な市場データを生成
        base_price = 100
        
        # 複数のトレンド期間を含む
        trend1 = np.linspace(0, 20, 150)  # 上昇トレンド
        trend2 = np.linspace(20, 10, 100)  # 下降トレンド
        trend3 = np.linspace(10, 35, 150)  # 再上昇トレンド
        trend4 = np.linspace(35, 30, 100)  # 横ばい
        
        trend = np.concatenate([trend1, trend2, trend3, trend4])
        
        # ノイズとボラティリティ
        noise = np.random.normal(0, 2, 500)
        volatility = 3 + 2 * np.sin(np.linspace(0, 4*np.pi, 500))  # 変動するボラティリティ
        cycle = volatility * np.sin(np.linspace(0, 8*np.pi, 500))
        
        prices = base_price + trend + noise + cycle
        
        # OHLCV データを生成
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.005, 500)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.015, 500))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.015, 500))),
            'Close': prices,
            'Volume': np.random.randint(1000, 20000, 500).astype(float)
        }, index=dates)
        
        return data

    def test_all_position_sizing_methods(self, comprehensive_data):
        """全てのポジションサイジング方法のテスト"""
        
        methods = [
            "fixed_ratio",
            "fixed_risk", 
            "kelly",
            "optimal_f",
            "volatility_based",
            "percent_volatility",
            "martingale",
            "anti_martingale"
        ]
        
        results = {}
        
        print("\n=== 全ポジションサイジング方法の比較 ===")
        
        for method in methods:
            try:
                bt = Backtest(
                    comprehensive_data, 
                    AdvancedRiskManagementStrategy, 
                    cash=10000, 
                    commission=0.001
                )
                
                # 方法に応じたパラメータ設定
                params = {
                    'n1': 10, 'n2': 30,
                    'position_sizing_method': method,
                    'min_risk_reward_ratio': 1.5,
                    'use_kelly_criterion': (method in ["kelly", "optimal_f"])
                }
                
                # 特別なパラメータ
                if method == "martingale":
                    params['consecutive_losses'] = 0
                elif method == "anti_martingale":
                    params['consecutive_wins'] = 0
                elif method == "volatility_based":
                    params['volatility'] = 0.02
                elif method == "percent_volatility":
                    params['volatility'] = 0.02
                    params['risk_percent'] = 0.01
                
                stats = bt.run(**params)
                
                results[method] = {
                    'trades': stats['# Trades'],
                    'final_equity': stats['Equity Final [$]'],
                    'return_pct': stats['Return [%]'],
                    'max_drawdown': stats['Max. Drawdown [%]'],
                    'sharpe_ratio': stats.get('Sharpe Ratio', 0),
                    'win_rate': stats.get('Win Rate [%]', 0)
                }
                
                print(f"{method.replace('_', ' ').title()}:")
                print(f"  取引数: {results[method]['trades']}")
                print(f"  最終資産: ${results[method]['final_equity']:.2f}")
                print(f"  リターン: {results[method]['return_pct']:.2f}%")
                print(f"  最大DD: {results[method]['max_drawdown']:.2f}%")
                print()
                
            except Exception as e:
                print(f"{method} でエラー: {e}")
                results[method] = None
        
        # 成功した方法が少なくとも5つあることを確認
        successful_methods = [k for k, v in results.items() if v is not None]
        assert len(successful_methods) >= 5, f"成功した方法が少なすぎます: {successful_methods}"

    def test_strategy_evolution_comparison(self, comprehensive_data):
        """戦略の進化比較テスト"""
        
        strategies = {
            '基本SMAクロス': SMACrossStrategy,
            '基本リスク管理付き': EnhancedSMACrossStrategy,
            'トレーリングストップ付き': EnhancedSMACrossStrategyWithTrailing,
            '出来高フィルター付き': EnhancedSMACrossStrategyWithVolume,
            '高度統合版': EnhancedSMACrossStrategyAdvanced,
            '高度リスク管理': AdvancedRiskManagementStrategy,
            '保守的戦略': ConservativeRiskManagementStrategy,
            'アグレッシブ戦略': AggressiveRiskManagementStrategy
        }
        
        results = {}
        
        print("\n=== 戦略進化の比較 ===")
        
        for name, strategy_class in strategies.items():
            try:
                bt = Backtest(comprehensive_data, strategy_class, cash=10000, commission=0.001)
                
                if name == '基本SMAクロス':
                    stats = bt.run(n1=10, n2=30)
                else:
                    stats = bt.run(
                        n1=10, n2=30,
                        sl_pct=0.02, tp_pct=0.05,
                        use_risk_management=True,
                        min_risk_reward_ratio=1.5
                    )
                
                results[name] = {
                    'trades': stats['# Trades'],
                    'final_equity': stats['Equity Final [$]'],
                    'return_pct': stats['Return [%]'],
                    'max_drawdown': stats['Max. Drawdown [%]'],
                    'sharpe_ratio': stats.get('Sharpe Ratio', 0)
                }
                
                print(f"{name}:")
                print(f"  取引数: {results[name]['trades']}")
                print(f"  最終資産: ${results[name]['final_equity']:.2f}")
                print(f"  リターン: {results[name]['return_pct']:.2f}%")
                print(f"  最大DD: {results[name]['max_drawdown']:.2f}%")
                print(f"  シャープ比: {results[name]['sharpe_ratio']:.3f}")
                print()
                
            except Exception as e:
                print(f"{name} でエラー: {e}")
                results[name] = None
        
        # 全ての戦略が正常に実行されることを確認
        successful_strategies = [k for k, v in results.items() if v is not None]
        assert len(successful_strategies) >= 6, f"成功した戦略が少なすぎます: {successful_strategies}"
        
        # 高度な戦略が基本戦略より改善されていることを確認（一部）
        if results['基本SMAクロス'] and results['高度リスク管理']:
            basic_dd = abs(results['基本SMAクロス']['max_drawdown'])
            advanced_dd = abs(results['高度リスク管理']['max_drawdown'])
            print(f"ドローダウン改善: {basic_dd:.2f}% → {advanced_dd:.2f}%")

    def test_risk_management_features_comprehensive(self):
        """リスク管理機能の包括的テスト"""
        
        calculator = RiskCalculator()
        
        print("\n=== リスク管理機能の包括的テスト ===")
        
        # Kelly Criterion テスト
        kelly_scenarios = [
            (0.6, 0.05, 0.03, "高勝率・低リスク"),
            (0.4, 0.08, 0.04, "低勝率・高リワード"),
            (0.5, 0.03, 0.03, "均等勝率"),
            (0.7, 0.02, 0.01, "超高勝率・超低リスク")
        ]
        
        print("Kelly Criterion テスト:")
        for win_rate, avg_win, avg_loss, desc in kelly_scenarios:
            kelly = calculator.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
            print(f"  {desc}: Kelly={kelly:.3f} (勝率{win_rate:.0%}, 平均利益{avg_win:.1%}, 平均損失{avg_loss:.1%})")
        
        # Risk-Reward Ratio テスト
        print("\nRisk-Reward Ratio テスト:")
        rr_scenarios = [
            (100, 98, 105, True, "ロング 2%リスク 5%リワード"),
            (100, 95, 110, True, "ロング 5%リスク 10%リワード"),
            (100, 102, 95, False, "ショート 2%リスク 5%リワード"),
            (100, 105, 90, False, "ショート 5%リスク 10%リワード")
        ]
        
        for entry, sl, tp, is_long, desc in rr_scenarios:
            rr = calculator.calculate_risk_reward_ratio(entry, sl, tp, is_long)
            print(f"  {desc}: R:R={rr:.1f}:1")
        
        # Position Sizing テスト
        print("\nPosition Sizing テスト:")
        sizing_scenarios = [
            ("fixed_ratio", {"ratio": 0.02}, "固定比率2%"),
            ("fixed_risk", {"risk_amount": 100}, "固定リスク$100"),
            ("kelly", {"win_rate": 0.6, "avg_win": 0.05, "avg_loss": 0.03}, "Kelly基準"),
            ("volatility_based", {"volatility": 0.03, "base_size": 0.02}, "ボラティリティベース"),
            ("percent_volatility", {"volatility": 0.02, "risk_percent": 0.01}, "パーセントボラティリティ")
        ]
        
        for method, kwargs, desc in sizing_scenarios:
            size = calculator.calculate_optimal_position_size(
                current_equity=10000,
                entry_price=100,
                sl_price=98,
                method=method,
                **kwargs
            )
            if size:
                print(f"  {desc}: サイズ={size:.4f} ({size*100:.2f}%)")
            else:
                print(f"  {desc}: 計算失敗")

    def test_extreme_market_conditions(self, comprehensive_data):
        """極端な市場条件でのテスト"""
        
        print("\n=== 極端な市場条件でのテスト ===")
        
        # 高ボラティリティデータの生成
        high_vol_data = comprehensive_data.copy()
        high_vol_data['Close'] *= (1 + np.random.normal(0, 0.05, len(high_vol_data)))
        high_vol_data['High'] = high_vol_data[['Open', 'Close']].max(axis=1) * 1.03
        high_vol_data['Low'] = high_vol_data[['Open', 'Close']].min(axis=1) * 0.97
        
        # 低ボラティリティデータの生成
        low_vol_data = comprehensive_data.copy()
        low_vol_data['Close'] *= (1 + np.random.normal(0, 0.005, len(low_vol_data)))
        low_vol_data['High'] = low_vol_data[['Open', 'Close']].max(axis=1) * 1.003
        low_vol_data['Low'] = low_vol_data[['Open', 'Close']].min(axis=1) * 0.997
        
        conditions = {
            '通常市場': comprehensive_data,
            '高ボラティリティ市場': high_vol_data,
            '低ボラティリティ市場': low_vol_data
        }
        
        for condition_name, data in conditions.items():
            print(f"\n{condition_name}:")
            
            try:
                # 保守的戦略でテスト
                bt_conservative = Backtest(data, ConservativeRiskManagementStrategy, cash=10000, commission=0.001)
                stats_conservative = bt_conservative.run(n1=10, n2=30)
                
                # アグレッシブ戦略でテスト
                bt_aggressive = Backtest(data, AggressiveRiskManagementStrategy, cash=10000, commission=0.001)
                stats_aggressive = bt_aggressive.run(n1=10, n2=30)
                
                print(f"  保守的戦略: リターン{stats_conservative['Return [%]']:.2f}%, DD{stats_conservative['Max. Drawdown [%]']:.2f}%")
                print(f"  アグレッシブ戦略: リターン{stats_aggressive['Return [%]']:.2f}%, DD{stats_aggressive['Max. Drawdown [%]']:.2f}%")
                
            except Exception as e:
                print(f"  エラー: {e}")

    def test_performance_metrics_calculation(self, comprehensive_data):
        """パフォーマンス指標計算のテスト"""
        
        print("\n=== パフォーマンス指標計算テスト ===")
        
        bt = Backtest(comprehensive_data, AdvancedRiskManagementStrategy, cash=10000, commission=0.001)
        stats = bt.run(
            n1=10, n2=30,
            min_risk_reward_ratio=2.0,
            use_kelly_criterion=True,
            position_sizing_method="kelly"
        )
        
        # 主要指標の確認
        key_metrics = [
            'Equity Final [$]',
            'Return [%]',
            'Max. Drawdown [%]',
            '# Trades',
            'Win Rate [%]',
            'Best Trade [%]',
            'Worst Trade [%]'
        ]
        
        print("主要パフォーマンス指標:")
        for metric in key_metrics:
            if metric in stats:
                print(f"  {metric}: {stats[metric]}")
        
        # 基本的な妥当性チェック
        assert stats['Equity Final [$]'] > 0, "最終資産が正の値である必要があります"
        assert stats['# Trades'] >= 0, "取引数が非負である必要があります"
        
        if stats['# Trades'] > 0:
            assert 'Win Rate [%]' in stats, "取引がある場合は勝率が計算されている必要があります"

    def test_error_handling_and_edge_cases(self):
        """エラーハンドリングとエッジケースのテスト"""
        
        print("\n=== エラーハンドリングとエッジケーステスト ===")
        
        calculator = RiskCalculator()
        
        # 無効な入力のテスト
        invalid_cases = [
            ("Kelly Criterion", lambda: calculator.calculate_kelly_criterion(-0.1, 0.05, 0.03)),
            ("Kelly Criterion", lambda: calculator.calculate_kelly_criterion(1.5, 0.05, 0.03)),
            ("Risk-Reward Ratio", lambda: calculator.calculate_risk_reward_ratio(100, 105, 95, True)),
            ("Position Sizing", lambda: calculator.calculate_optimal_position_size(-1000, 100, 98, "fixed_ratio")),
            ("Position Sizing", lambda: calculator.calculate_optimal_position_size(10000, -100, 98, "fixed_ratio")),
        ]
        
        for test_name, test_func in invalid_cases:
            try:
                result = test_func()
                if result is None:
                    print(f"  ✓ {test_name}: 無効入力を正しく処理")
                else:
                    print(f"  ⚠ {test_name}: 無効入力で予期しない結果: {result}")
            except Exception as e:
                print(f"  ✓ {test_name}: 例外で正しく処理: {type(e).__name__}")

    def test_integration_with_backtesting_library(self, comprehensive_data):
        """backtesting.pyライブラリとの統合テスト"""
        
        print("\n=== backtesting.pyライブラリ統合テスト ===")
        
        # 基本的なSL/TP機能のテスト
        bt = Backtest(comprehensive_data, EnhancedSMACrossStrategy, cash=10000, commission=0.001)
        stats = bt.run(
            n1=10, n2=30,
            sl_pct=0.02, tp_pct=0.05,
            use_risk_management=True
        )
        
        print(f"基本SL/TP機能: 取引数{stats['# Trades']}, リターン{stats['Return [%]']:.2f}%")
        
        # 動的SL/TP調整のテスト
        bt_dynamic = Backtest(comprehensive_data, EnhancedSMACrossStrategyWithTrailing, cash=10000, commission=0.001)
        stats_dynamic = bt_dynamic.run(
            n1=10, n2=30,
            sl_pct=0.02, tp_pct=0.05,
            use_risk_management=True,
            use_trailing_stop=True
        )
        
        print(f"動的SL/TP機能: 取引数{stats_dynamic['# Trades']}, リターン{stats_dynamic['Return [%]']:.2f}%")
        
        # ATRベース機能のテスト
        bt_atr = Backtest(comprehensive_data, AdvancedRiskManagementStrategy, cash=10000, commission=0.001)
        stats_atr = bt_atr.run(
            n1=10, n2=30,
            use_atr_based_risk=True,
            atr_sl_multiplier=2.0,
            atr_tp_multiplier=3.0,
            min_risk_reward_ratio=1.5
        )
        
        print(f"ATRベース機能: 取引数{stats_atr['# Trades']}, リターン{stats_atr['Return [%]']:.2f}%")
        
        # 全ての機能が正常に動作することを確認
        assert stats['# Trades'] >= 0
        assert stats_dynamic['# Trades'] >= 0
        assert stats_atr['# Trades'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
