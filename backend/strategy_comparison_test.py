#!/usr/bin/env python3
"""
戦略比較テスト実行スクリプト

注意: このスクリプトは独自実装のStrategyExecutorに依存していましたが、
backtesting.pyライブラリへの統一により無効化されました。

新しい戦略比較テストは以下を参照してください:
- backend/tests/accuracy/test_backtest_accuracy.py
- backend/tests/integration/test_strategy_switching.py

このファイルは参考用として残されています。
"""

import sys
print("このスクリプトは無効化されました。")
print("新しいバックテストシステムについては以下を参照してください:")
print("- backend/tests/accuracy/test_backtest_accuracy.py")
print("- backend/tests/integration/test_strategy_switching.py")
sys.exit(0)

# 以下は無効化されたコード（参考用）
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
# from backtest.engine.strategy_executor import StrategyExecutor  # 削除済み

def create_btc_data():
    """BTCテストデータの作成"""
    dates = pd.date_range('2024-01-01', periods=90, freq='1D')
    base_price = 45000
    prices = [base_price]
    
    # リアルなBTC価格パターン（上昇トレンド + ボラティリティ）
    np.random.seed(42)  # 再現性のため
    for i in range(89):
        daily_return = np.random.normal(0.002, 0.03)  # 年率約73%、30%ボラティリティ
        prices.append(prices[-1] * (1 + daily_return))
    
    data = []
    for i, close_price in enumerate(prices):
        open_price = close_price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
        volume = np.random.uniform(500, 1500)
        
        data.append({
            'Open': round(open_price, 2),
            'High': round(high_price, 2),
            'Low': round(low_price, 2),
            'Close': round(close_price, 2),
            'Volume': volume
        })
    
    return pd.DataFrame(data, index=dates)

def get_strategies():
    """戦略設定を取得"""
    return {
        'SMA_Cross': {
            'name': 'SMA Cross Strategy',
            'indicators': [
                {'name': 'SMA', 'params': {'period': 20}},
                {'name': 'SMA', 'params': {'period': 50}}
            ],
            'entry_rules': [
                {'condition': 'SMA(close, 20) > SMA(close, 50)'}
            ],
            'exit_rules': [
                {'condition': 'SMA(close, 20) < SMA(close, 50)'}
            ]
        },
        'RSI_Strategy': {
            'name': 'RSI Strategy',
            'indicators': [
                {'name': 'RSI', 'params': {'period': 14}}
            ],
            'entry_rules': [
                {'condition': 'RSI(close, 14) < 30'}
            ],
            'exit_rules': [
                {'condition': 'RSI(close, 14) > 70'}
            ]
        },
        'EMA_Cross': {
            'name': 'EMA Cross Strategy',
            'indicators': [
                {'name': 'EMA', 'params': {'period': 12}},
                {'name': 'EMA', 'params': {'period': 26}}
            ],
            'entry_rules': [
                {'condition': 'EMA(close, 12) > EMA(close, 26)'}
            ],
            'exit_rules': [
                {'condition': 'EMA(close, 12) < EMA(close, 26)'}
            ]
        },
        'Buy_Hold': {
            'name': 'Buy and Hold',
            'indicators': [],
            'entry_rules': [
                {'condition': 'close > 0'}  # 最初に買って保持
            ],
            'exit_rules': []  # 売らない
        }
    }

def main():
    """メイン実行関数"""
    # データ作成
    data = create_btc_data()
    print('=== BTC テストデータ ===')
    print(f'期間: {data.index[0].date()} - {data.index[-1].date()}')
    print(f'開始価格: ${data.iloc[0]["Close"]:,.2f}')
    print(f'終了価格: ${data.iloc[-1]["Close"]:,.2f}')
    print(f'期間リターン: {((data.iloc[-1]["Close"] / data.iloc[0]["Close"]) - 1) * 100:.2f}%')
    print()

    # 戦略比較
    strategies = get_strategies()
    results = {}
    print('=== 戦略比較結果 ===')

    for strategy_name, strategy_config in strategies.items():
        try:
            executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)
            result = executor.run_backtest(data, strategy_config)
            results[strategy_name] = result
            
            print(f'{strategy_name}:')
            print(f'  総リターン: {result["total_return"] * 100:.2f}%')
            print(f'  最終資産: ${result["final_equity"]:,.2f}')
            print(f'  取引回数: {result["total_trades"]}')
            print(f'  勝率: {result["win_rate"] * 100:.1f}%')
            print(f'  シャープレシオ: {result["sharpe_ratio"]:.3f}')
            print(f'  最大ドローダウン: {result["max_drawdown"] * 100:.2f}%')
            print()
        except Exception as e:
            print(f'{strategy_name}: エラー - {e}')
            print()

    # 結果分析
    if results:
        print('=== 結果分析 ===')
        
        # 最良戦略（総リターン）
        best_return = max(results.items(), key=lambda x: x[1]['total_return'])
        print(f'最高リターン: {best_return[0]} ({best_return[1]["total_return"] * 100:.2f}%)')
        
        # 最良戦略（シャープレシオ）
        best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])
        print(f'最高シャープレシオ: {best_sharpe[0]} ({best_sharpe[1]["sharpe_ratio"]:.3f})')
        
        # 最小ドローダウン
        min_dd = min(results.items(), key=lambda x: x[1]['max_drawdown'])
        print(f'最小ドローダウン: {min_dd[0]} ({min_dd[1]["max_drawdown"] * 100:.2f}%)')
        
        # 最多取引
        max_trades = max(results.items(), key=lambda x: x[1]['total_trades'])
        print(f'最多取引: {max_trades[0]} ({max_trades[1]["total_trades"]}回)')
        
        print()
        print('=== 戦略特性分析 ===')
        
        # 戦略タイプ別分析
        trend_following = ['SMA_Cross', 'EMA_Cross']
        mean_reversion = ['RSI_Strategy']
        passive = ['Buy_Hold']
        
        for category, strategy_list in [
            ('トレンドフォロー', trend_following),
            ('平均回帰', mean_reversion),
            ('パッシブ', passive)
        ]:
            category_results = {k: v for k, v in results.items() if k in strategy_list}
            if category_results:
                avg_return = np.mean([r['total_return'] for r in category_results.values()])
                print(f'{category}戦略平均リターン: {avg_return * 100:.2f}%')

if __name__ == "__main__":
    main()
