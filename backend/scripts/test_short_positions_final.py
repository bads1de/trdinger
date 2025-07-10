#!/usr/bin/env python3
"""
ショートポジション最終確認テスト

修正されたConditionEvaluatorとStrategyFactoryで
実際にショートポジションが発生することを確認します。
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.core.services.auto_strategy.models.ga_config import GAConfig
from backtesting import Backtest


def create_test_data():
    """テスト用の市場データを作成"""
    # 1週間分の時間足データ
    dates = pd.date_range(start="2024-01-01", end="2024-01-07", freq="1h")
    
    # トレンド変化のあるデータを作成
    np.random.seed(42)
    base_price = 50000
    prices = []
    
    for i in range(len(dates)):
        # 前半は上昇トレンド、後半は下降トレンド
        if i < len(dates) // 2:
            trend = 1.001  # 上昇
        else:
            trend = 0.999  # 下降
            
        if i == 0:
            price = base_price
        else:
            price = prices[-1] * trend * (1 + np.random.normal(0, 0.01))
        
        prices.append(price)
    
    # OHLCV データを作成
    data = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # High/Lowの調整
    data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
    data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
    
    return data


def test_short_position_strategy():
    """ショートポジション戦略のテスト"""
    print("🔍 ショートポジション戦略テスト開始")
    print("=" * 50)
    
    # テストデータを作成
    data = create_test_data()
    print(f"📊 テストデータ: {len(data)}行")
    print(f"   期間: {data.index[0]} - {data.index[-1]}")
    print(f"   価格範囲: {data['Close'].min():.2f} - {data['Close'].max():.2f}")
    
    # GA設定
    ga_config = GAConfig.create_fast()
    generator = RandomGeneGenerator(ga_config, enable_smart_generation=True)
    factory = StrategyFactory()
    
    # 複数の戦略をテスト
    test_results = []
    
    for i in range(5):
        print(f"\n--- 戦略 {i+1} ---")
        
        # 戦略遺伝子を生成
        gene = generator.generate_random_gene()
        
        print(f"ロング条件数: {len(gene.long_entry_conditions)}")
        print(f"ショート条件数: {len(gene.short_entry_conditions)}")
        
        # 条件の詳細を表示
        for j, cond in enumerate(gene.long_entry_conditions):
            print(f"  ロング{j+1}: {cond.left_operand} {cond.operator} {cond.right_operand}")
        for j, cond in enumerate(gene.short_entry_conditions):
            print(f"  ショート{j+1}: {cond.left_operand} {cond.operator} {cond.right_operand}")
        
        try:
            # 戦略クラスを作成
            strategy_class = factory.create_strategy_class(gene)
            
            # バックテストを実行
            bt = Backtest(data, strategy_class, cash=100000, commission=0.001)
            result = bt.run()
            
            # 結果を分析
            trades = result._trades if hasattr(result, '_trades') else []
            long_trades = [t for t in trades if t.get('Size', 0) > 0]
            short_trades = [t for t in trades if t.get('Size', 0) < 0]
            
            test_result = {
                'strategy_id': i + 1,
                'total_trades': len(trades),
                'long_trades': len(long_trades),
                'short_trades': len(short_trades),
                'return': result.get('Return [%]', 0),
                'success': True
            }
            
            print(f"  📈 総取引数: {len(trades)}")
            print(f"  📈 ロング取引: {len(long_trades)}")
            print(f"  📉 ショート取引: {len(short_trades)}")
            print(f"  💰 リターン: {result.get('Return [%]', 0):.2f}%")
            
            if len(short_trades) > 0:
                print(f"  ✅ ショートポジション確認！")
            
        except Exception as e:
            print(f"  ❌ エラー: {e}")
            test_result = {
                'strategy_id': i + 1,
                'total_trades': 0,
                'long_trades': 0,
                'short_trades': 0,
                'return': 0,
                'success': False,
                'error': str(e)
            }
        
        test_results.append(test_result)
    
    # 結果のサマリー
    print(f"\n📊 テスト結果サマリー")
    print("=" * 50)
    
    successful_tests = [r for r in test_results if r['success']]
    total_short_trades = sum(r['short_trades'] for r in successful_tests)
    total_long_trades = sum(r['long_trades'] for r in successful_tests)
    strategies_with_shorts = len([r for r in successful_tests if r['short_trades'] > 0])
    
    print(f"成功した戦略: {len(successful_tests)}/5")
    print(f"ショートポジションを持つ戦略: {strategies_with_shorts}/5")
    print(f"総ロング取引: {total_long_trades}")
    print(f"総ショート取引: {total_short_trades}")
    
    if total_short_trades > 0:
        print(f"\n🎉 ショートポジション問題が解決されました！")
        print(f"   修正されたConditionEvaluatorとStrategyFactoryが正常に動作しています。")
    else:
        print(f"\n⚠️  まだショートポジションが発生していません。")
        print(f"   さらなる調査が必要です。")
    
    return test_results


def main():
    """メイン実行"""
    try:
        test_short_position_strategy()
        print(f"\n✅ ショートポジション最終確認テスト完了")
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
