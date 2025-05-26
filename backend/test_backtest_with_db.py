#!/usr/bin/env python3
"""
データベースデータを使用したバックテストのテスト
"""
import json
import os
from datetime import datetime, timedelta

# SQLite用の設定
os.environ["DATABASE_URL"] = "sqlite:///./trdinger_test.db"

from backtest_runner import run_backtest

def test_backtest_with_real_data():
    """
    実際のデータベースデータを使用してバックテストをテスト
    """
    print("=== データベースデータを使用したバックテストテスト ===")

    # テスト用の戦略設定
    strategy_config = {
        "strategy": {
            "id": "test_sma_cross",
            "name": "SMA Cross Strategy",
            "target_pair": "BTC/USD:BTC",
            "indicators": [
                {
                    "name": "SMA",
                    "params": {"period": 10}
                },
                {
                    "name": "SMA",
                    "params": {"period": 20}
                }
            ],
            "entry_rules": [
                {"condition": "SMA(close, 10) > SMA(close, 20)"}
            ],
            "exit_rules": [
                {"condition": "SMA(close, 10) < SMA(close, 20)"}
            ]
        },
        "start_date": "2024-12-01T00:00:00Z",
        "end_date": "2025-05-01T00:00:00Z",
        "timeframe": "1d",
        "initial_capital": 100000,
        "commission_rate": 0.001
    }

    try:
        # バックテストを実行
        print("バックテスト実行中...")
        result = run_backtest(strategy_config)

        if 'error' in result:
            print(f"❌ バックテストエラー: {result['error']}")
            return

        # 結果を表示
        print("✅ バックテスト完了!")
        print(f"📊 パフォーマンス結果:")
        print(f"  総リターン: {result['total_return']:.2%}")
        print(f"  シャープレシオ: {result['sharpe_ratio']:.3f}")
        print(f"  最大ドローダウン: {result['max_drawdown']:.2%}")
        print(f"  勝率: {result['win_rate']:.2%}")
        print(f"  プロフィットファクター: {result['profit_factor']:.3f}")
        print(f"  総取引数: {result['total_trades']}")
        print(f"  勝ちトレード: {result['winning_trades']}")
        print(f"  負けトレード: {result['losing_trades']}")
        print(f"  平均利益: ${result['avg_win']:.2f}")
        print(f"  平均損失: ${result['avg_loss']:.2f}")
        final_equity = result.get('final_equity', result.get('equity_curve', [{}])[-1].get('equity', 0) if result.get('equity_curve') else 0)
        print(f"  最終資産: ${final_equity:.2f}")

        # 取引履歴の一部を表示
        if result['trade_history']:
            print(f"\n📈 取引履歴（最初の5件）:")
            for i, trade in enumerate(result['trade_history'][:5]):
                print(f"  {i+1}. {trade['timestamp'][:10]} {trade['type'].upper()} "
                      f"${trade['price']:.2f} x {trade['quantity']:.6f} "
                      f"PnL: ${trade['pnl']:.2f}")

        # 資産曲線の一部を表示
        if result['equity_curve']:
            print(f"\n📈 資産曲線（最後の5件）:")
            for point in result['equity_curve'][-5:]:
                print(f"  {point['timestamp'][:10]}: ${point['equity']:.2f}")

        print("\n🎉 データベース統合バックテストが正常に動作しています！")

    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_backtest_with_real_data()
