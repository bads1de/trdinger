#!/usr/bin/env python3
"""
サンプルデータを使用した自動戦略生成テスト
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_ga_with_sample_data():
    """サンプルデータでGA機能をテスト"""
    print("🧬 サンプルデータでの自動戦略生成テスト")
    print("=" * 60)
    
    try:
        # 1. 必要なモジュールのインポート
        print("1. モジュールインポート中...")
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene, IndicatorGene, Condition
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
        from backtesting import Backtest
        print("  ✅ インポート完了")
        
        # 2. サンプルデータ生成
        print("\n2. サンプルデータ生成中...")
        def generate_sample_data(days=100):
            """サンプルのBTC価格データを生成"""
            dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
            np.random.seed(42)  # 再現性のため
            
            # より現実的な価格変動を生成
            initial_price = 50000
            daily_returns = np.random.normal(0.001, 0.02, days)  # 平均0.1%、標準偏差2%
            
            # 価格を計算
            prices = [initial_price]
            for ret in daily_returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # OHLCV データを生成
            data = []
            for i, (date, price) in enumerate(zip(dates, prices)):
                daily_volatility = np.random.uniform(0.005, 0.015)  # 0.5-1.5%の日中変動
                
                high = price * (1 + daily_volatility)
                low = price * (1 - daily_volatility)
                
                if i == 0:
                    open_price = price
                else:
                    open_price = prices[i-1]
                close_price = price
                
                volume = np.random.uniform(800000, 1200000)
                
                data.append({
                    'Open': open_price,
                    'High': high,
                    'Low': low,
                    'Close': close_price,
                    'Volume': volume
                })
            
            df = pd.DataFrame(data, index=dates)
            return df
        
        sample_data = generate_sample_data(100)
        print(f"  ✅ サンプルデータ生成: {len(sample_data)}日分")
        print(f"    価格範囲: ${sample_data['Close'].min():.0f} - ${sample_data['Close'].max():.0f}")
        
        # 3. 手動戦略の作成とテスト
        print("\n3. 手動戦略の作成とテスト...")
        manual_gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)
            ],
            entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="SMA_20")
            ],
            exit_conditions=[
                Condition(left_operand="close", operator="<", right_operand="SMA_20")
            ]
        )
        
        factory = StrategyFactory()
        
        # 妥当性チェック
        is_valid, errors = factory.validate_gene(manual_gene)
        print(f"  妥当性チェック: {is_valid}")
        if not is_valid:
            print(f"  エラー: {errors}")
            return False
        
        # 戦略クラス生成
        try:
            strategy_class = factory.create_strategy_class(manual_gene)
            print(f"  ✅ 戦略クラス生成成功: {strategy_class.__name__}")
        except Exception as e:
            print(f"  ❌ 戦略クラス生成失敗: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 4. バックテスト実行
        print("\n4. バックテスト実行中...")
        try:
            bt = Backtest(
                sample_data,
                strategy_class,
                cash=100000,
                commission=0.001,
                exclusive_orders=True,
                trade_on_close=True
            )
            
            stats = bt.run()
            print(f"  ✅ バックテスト実行成功")
            print(f"    総リターン: {stats['Return [%]']:.2f}%")
            print(f"    取引回数: {stats['# Trades']}")
            print(f"    勝率: {stats['Win Rate [%]']:.2f}%")
            print(f"    シャープレシオ: {stats['Sharpe Ratio']:.4f}")
            print(f"    最大ドローダウン: {stats['Max. Drawdown [%]']:.2f}%")
            
            # 取引があったかチェック
            if stats['# Trades'] > 0:
                print(f"  🎉 戦略が実際に取引を実行しました！")
                return True
            else:
                print(f"  ⚠️ 戦略が取引を実行しませんでした")
                
        except Exception as e:
            print(f"  ❌ バックテスト実行失敗: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 5. 複数の戦略パターンをテスト
        print("\n5. 複数の戦略パターンをテスト...")
        
        test_strategies = [
            {
                "name": "RSI Oversold/Overbought",
                "gene": StrategyGene(
                    indicators=[
                        IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)
                    ],
                    entry_conditions=[
                        Condition(left_operand="RSI_14", operator="<", right_operand=30)
                    ],
                    exit_conditions=[
                        Condition(left_operand="RSI_14", operator=">", right_operand=70)
                    ]
                )
            },
            {
                "name": "SMA Crossover",
                "gene": StrategyGene(
                    indicators=[
                        IndicatorGene(type="SMA", parameters={"period": 10}, enabled=True),
                        IndicatorGene(type="SMA", parameters={"period": 30}, enabled=True)
                    ],
                    entry_conditions=[
                        Condition(left_operand="SMA_10", operator=">", right_operand="SMA_30")
                    ],
                    exit_conditions=[
                        Condition(left_operand="SMA_10", operator="<", right_operand="SMA_30")
                    ]
                )
            }
        ]
        
        successful_strategies = 0
        
        for strategy_info in test_strategies:
            print(f"\n  テスト中: {strategy_info['name']}")
            try:
                test_strategy_class = factory.create_strategy_class(strategy_info['gene'])
                test_bt = Backtest(
                    sample_data,
                    test_strategy_class,
                    cash=100000,
                    commission=0.001,
                    exclusive_orders=True,
                    trade_on_close=True
                )
                
                test_stats = test_bt.run()
                trades = test_stats['# Trades']
                returns = test_stats['Return [%]']
                
                print(f"    取引回数: {trades}, リターン: {returns:.2f}%")
                
                if trades > 0:
                    successful_strategies += 1
                    print(f"    ✅ 成功")
                else:
                    print(f"    ⚠️ 取引なし")
                    
            except Exception as e:
                print(f"    ❌ エラー: {e}")
        
        print(f"\n📊 結果サマリー:")
        print(f"  テスト戦略数: {len(test_strategies)}")
        print(f"  成功戦略数: {successful_strategies}")
        print(f"  成功率: {successful_strategies/len(test_strategies)*100:.1f}%")
        
        if successful_strategies > 0:
            print(f"\n🎉 自動戦略生成機能は正常に動作しています！")
            print(f"   実際に取引を実行する戦略が生成されました。")
            return True
        else:
            print(f"\n⚠️ 戦略は生成されましたが、取引が実行されませんでした。")
            print(f"   条件やデータを調整する必要があります。")
            return False
        
    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_ga_with_sample_data()
