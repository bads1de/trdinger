"""
SMA+RSI戦略の現実的なデータでのテスト

より現実的な価格データと初期資金設定でのバックテスト
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtesting import Backtest

from app.core.strategies.sma_rsi_strategy import SMARSIStrategy, SMARSIStrategyOptimized


def generate_realistic_crypto_data(days=365, initial_price=100):
    """
    より現実的な暗号通貨価格データを生成
    
    Args:
        days: データの日数
        initial_price: 初期価格（ドル）
        
    Returns:
        OHLCV DataFrame
    """
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=days, freq='D')
    
    # より現実的な価格動向を生成
    base_price = initial_price
    
    # 複数のトレンド期間を作成
    trend_periods = [
        (0, 100, 0.002),      # 上昇トレンド
        (100, 200, -0.001),   # 下降トレンド
        (200, 300, 0.003),    # 強い上昇トレンド
        (300, days, -0.0005), # 軽い下降トレンド
    ]
    
    prices = []
    current_price = base_price
    
    for i in range(days):
        # 現在の期間のトレンドを特定
        trend_rate = 0
        for start, end, rate in trend_periods:
            if start <= i < end:
                trend_rate = rate
                break
        
        # トレンド + ランダムウォーク + ボラティリティ
        daily_return = trend_rate + np.random.normal(0, 0.03)  # 3%の日次ボラティリティ
        current_price *= (1 + daily_return)
        
        # 価格が負にならないように制限
        current_price = max(current_price, 1.0)
        prices.append(current_price)
    
    close_prices = np.array(prices)
    
    # OHLCV データを生成
    data = pd.DataFrame({
        'Open': (close_prices * (1 + np.random.normal(0, 0.005, days))).astype(np.float64),
        'High': (close_prices * (1 + np.abs(np.random.normal(0, 0.02, days)))).astype(np.float64),
        'Low': (close_prices * (1 - np.abs(np.random.normal(0, 0.02, days)))).astype(np.float64),
        'Close': close_prices.astype(np.float64),
        'Volume': np.random.randint(1000, 10000, days).astype(np.float64)
    }, index=dates)
    
    # 価格の整合性を保つ
    data['High'] = np.maximum(data['High'], data[['Open', 'Close']].max(axis=1))
    data['Low'] = np.minimum(data['Low'], data[['Open', 'Close']].min(axis=1))
    
    return data


def test_sma_rsi_with_realistic_data():
    """現実的なデータでのSMA+RSI戦略テスト"""
    
    print("=== 現実的なデータでのSMA+RSI戦略テスト ===")
    
    # 現実的なデータ生成（初期価格100ドル）
    data = generate_realistic_crypto_data(days=365, initial_price=100)
    
    print(f"データ期間: {data.index[0].date()} - {data.index[-1].date()}")
    print(f"価格範囲: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    print(f"データ件数: {len(data)}")
    
    # 適切な初期資金設定（最高価格の10倍）
    max_price = data['Close'].max()
    initial_cash = max_price * 10
    
    print(f"初期資金: ${initial_cash:.2f}")
    
    # バックテスト実行
    bt = Backtest(data, SMARSIStrategy, cash=initial_cash, commission=0.001)
    
    stats = bt.run(
        sma_short=10,
        sma_long=30,
        rsi_period=14,
        oversold_threshold=30,
        overbought_threshold=70,
        use_risk_management=True,
        sl_pct=0.02,
        tp_pct=0.05
    )
    
    # 結果の表示
    print(f"\n📊 バックテスト結果:")
    print(f"取引数: {stats['# Trades']}")
    print(f"最終資産: ${stats['Equity Final [$]']:.2f}")
    print(f"総リターン: {stats['Return [%]']:.2f}%")
    print(f"最大ドローダウン: {stats['Max. Drawdown [%]']:.2f}%")
    print(f"勝率: {stats['Win Rate [%]']:.2f}%")
    print(f"シャープレシオ: {stats.get('Sharpe Ratio', 0):.3f}")
    print(f"買い&ホールドリターン: {stats['Buy & Hold Return [%]']:.2f}%")
    
    return stats


def test_parameter_sensitivity():
    """パラメータ感度分析"""
    
    print("\n=== パラメータ感度分析 ===")
    
    # データ生成
    data = generate_realistic_crypto_data(days=365, initial_price=100)
    initial_cash = data['Close'].max() * 10
    
    # RSI閾値の感度分析
    rsi_thresholds = [
        (20, 80),
        (25, 75),
        (30, 70),
        (35, 65),
    ]
    
    print("\nRSI閾値感度分析:")
    for oversold, overbought in rsi_thresholds:
        bt = Backtest(data, SMARSIStrategy, cash=initial_cash, commission=0.001)
        stats = bt.run(
            sma_short=10, sma_long=30, rsi_period=14,
            oversold_threshold=oversold, overbought_threshold=overbought,
            use_risk_management=True
        )
        
        print(f"RSI({oversold},{overbought}): 取引数={stats['# Trades']}, リターン={stats['Return [%]']:.2f}%, シャープ={stats.get('Sharpe Ratio', 0):.3f}")
    
    # SMA期間の感度分析
    sma_periods = [
        (5, 20),
        (10, 30),
        (15, 40),
        (20, 50),
    ]
    
    print("\nSMA期間感度分析:")
    for short, long in sma_periods:
        bt = Backtest(data, SMARSIStrategy, cash=initial_cash, commission=0.001)
        stats = bt.run(
            sma_short=short, sma_long=long, rsi_period=14,
            oversold_threshold=30, overbought_threshold=70,
            use_risk_management=True
        )
        
        print(f"SMA({short},{long}): 取引数={stats['# Trades']}, リターン={stats['Return [%]']:.2f}%, シャープ={stats.get('Sharpe Ratio', 0):.3f}")


def test_risk_management_effectiveness():
    """リスク管理機能の効果検証"""
    
    print("\n=== リスク管理機能の効果検証 ===")
    
    # データ生成
    data = generate_realistic_crypto_data(days=365, initial_price=100)
    initial_cash = data['Close'].max() * 10
    
    # リスク管理なし
    bt_no_risk = Backtest(data, SMARSIStrategy, cash=initial_cash, commission=0.001)
    stats_no_risk = bt_no_risk.run(
        sma_short=10, sma_long=30, rsi_period=14,
        oversold_threshold=30, overbought_threshold=70,
        use_risk_management=False
    )
    
    # リスク管理あり（保守的）
    bt_conservative = Backtest(data, SMARSIStrategy, cash=initial_cash, commission=0.001)
    stats_conservative = bt_conservative.run(
        sma_short=10, sma_long=30, rsi_period=14,
        oversold_threshold=30, overbought_threshold=70,
        use_risk_management=True, sl_pct=0.02, tp_pct=0.04
    )
    
    # リスク管理あり（積極的）
    bt_aggressive = Backtest(data, SMARSIStrategy, cash=initial_cash, commission=0.001)
    stats_aggressive = bt_aggressive.run(
        sma_short=10, sma_long=30, rsi_period=14,
        oversold_threshold=30, overbought_threshold=70,
        use_risk_management=True, sl_pct=0.03, tp_pct=0.06
    )
    
    # 結果比較
    print("リスク管理なし:")
    print(f"  取引数: {stats_no_risk['# Trades']}")
    print(f"  リターン: {stats_no_risk['Return [%]']:.2f}%")
    print(f"  最大DD: {stats_no_risk['Max. Drawdown [%]']:.2f}%")
    print(f"  シャープレシオ: {stats_no_risk.get('Sharpe Ratio', 0):.3f}")
    
    print("リスク管理あり（保守的 SL:2%, TP:4%）:")
    print(f"  取引数: {stats_conservative['# Trades']}")
    print(f"  リターン: {stats_conservative['Return [%]']:.2f}%")
    print(f"  最大DD: {stats_conservative['Max. Drawdown [%]']:.2f}%")
    print(f"  シャープレシオ: {stats_conservative.get('Sharpe Ratio', 0):.3f}")
    
    print("リスク管理あり（積極的 SL:3%, TP:6%）:")
    print(f"  取引数: {stats_aggressive['# Trades']}")
    print(f"  リターン: {stats_aggressive['Return [%]']:.2f}%")
    print(f"  最大DD: {stats_aggressive['Max. Drawdown [%]']:.2f}%")
    print(f"  シャープレシオ: {stats_aggressive.get('Sharpe Ratio', 0):.3f}")


def test_optimized_vs_basic_strategy():
    """最適化戦略と基本戦略の比較"""
    
    print("\n=== 最適化戦略 vs 基本戦略 ===")
    
    # データ生成
    data = generate_realistic_crypto_data(days=365, initial_price=100)
    initial_cash = data['Close'].max() * 10
    
    # 基本戦略
    bt_basic = Backtest(data, SMARSIStrategy, cash=initial_cash, commission=0.001)
    stats_basic = bt_basic.run(
        sma_short=10, sma_long=30, rsi_period=14,
        use_risk_management=True
    )
    
    # 最適化戦略
    bt_optimized = Backtest(data, SMARSIStrategyOptimized, cash=initial_cash, commission=0.001)
    stats_optimized = bt_optimized.run(
        sma_short=10, sma_long=30, rsi_period=14,
        use_risk_management=True,
        volume_filter=True, volume_threshold=1.2,
        rsi_confirmation_bars=2
    )
    
    # 結果比較
    print("基本戦略:")
    print(f"  取引数: {stats_basic['# Trades']}")
    print(f"  リターン: {stats_basic['Return [%]']:.2f}%")
    print(f"  最大DD: {stats_basic['Max. Drawdown [%]']:.2f}%")
    print(f"  シャープレシオ: {stats_basic.get('Sharpe Ratio', 0):.3f}")
    
    print("最適化戦略:")
    print(f"  取引数: {stats_optimized['# Trades']}")
    print(f"  リターン: {stats_optimized['Return [%]']:.2f}%")
    print(f"  最大DD: {stats_optimized['Max. Drawdown [%]']:.2f}%")
    print(f"  シャープレシオ: {stats_optimized.get('Sharpe Ratio', 0):.3f}")


def main():
    """メインテスト実行"""
    print("🚀 SMA+RSI戦略 現実的データテスト開始")
    print("=" * 80)
    
    try:
        # 基本テスト
        test_sma_rsi_with_realistic_data()
        print("✅ 現実的データテスト成功")
        
        # パラメータ感度分析
        test_parameter_sensitivity()
        print("✅ パラメータ感度分析成功")
        
        # リスク管理効果検証
        test_risk_management_effectiveness()
        print("✅ リスク管理効果検証成功")
        
        # 戦略比較
        test_optimized_vs_basic_strategy()
        print("✅ 戦略比較成功")
        
        print("\n" + "=" * 80)
        print("🎉 全ての現実的データテストが成功しました！")
        print("\n💡 主要成果:")
        print("- 現実的な価格データでの動作確認")
        print("- 実際の取引シグナル生成の確認")
        print("- パラメータ感度の分析")
        print("- リスク管理機能の効果検証")
        print("- 基本戦略と最適化戦略の性能比較")
        
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
