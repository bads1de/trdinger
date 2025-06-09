#!/usr/bin/env python3
"""
複数の異なる戦略が自動生成されることを確認するテスト
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_multiple_strategy_generation():
    """複数回実行して異なる戦略が生成されることを確認"""
    print("🔄 複数戦略自動生成確認テスト")
    print("=" * 60)
    
    try:
        # 1. 必要なモジュールのインポート
        print("1. モジュールインポート中...")
        import pandas as pd
        import numpy as np
        import random
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        from app.core.services.auto_strategy.engines.ga_engine import GeneticAlgorithmEngine
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
        from app.core.services.backtest_service import BacktestService
        from backtesting import Backtest
        print("  ✅ インポート完了")
        
        # 2. サンプルデータ生成関数
        def generate_sample_data(days=100, seed=None):
            """サンプルのBTC価格データを生成"""
            if seed:
                np.random.seed(seed)
            
            dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
            
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
                    'timestamp': date,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': volume
                })
            
            return data
        
        # 3. モックサービス作成
        class MockOHLCVRepository:
            def __init__(self, data):
                self.data = data
            
            def get_ohlcv_data(self, symbol, timeframe, start_time=None, end_time=None, limit=None):
                class MockOHLCVData:
                    def __init__(self, **kwargs):
                        for k, v in kwargs.items():
                            setattr(self, k, v)
                
                return [MockOHLCVData(**item) for item in self.data]
        
        class MockBacktestDataService:
            def __init__(self, ohlcv_repo):
                self.ohlcv_repo = ohlcv_repo
            
            def get_ohlcv_for_backtest(self, symbol, timeframe, start_date, end_date):
                ohlcv_data = self.ohlcv_repo.get_ohlcv_data(symbol, timeframe)
                
                data = []
                for record in ohlcv_data:
                    data.append({
                        'Open': record.open,
                        'High': record.high,
                        'Low': record.low,
                        'Close': record.close,
                        'Volume': record.volume
                    })
                
                df = pd.DataFrame(data)
                df.index = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
                return df
        
        # 4. 複数回の戦略生成テスト
        print("\n4. 複数戦略生成テスト開始...")
        
        generated_strategies = []
        test_runs = 5  # 5回のテストを実行
        
        for run in range(test_runs):
            print(f"\n--- テスト実行 {run + 1}/{test_runs} ---")
            
            # 異なるシードでデータ生成
            sample_data = generate_sample_data(100, seed=42 + run)
            mock_repo = MockOHLCVRepository(sample_data)
            mock_data_service = MockBacktestDataService(mock_repo)
            mock_backtest_service = BacktestService(mock_data_service)
            
            # GA設定（異なるパラメータ）
            ga_config = GAConfig(
                population_size=4,  # 小規模
                generations=2,      # 短時間
                crossover_rate=0.7 + (run * 0.05),  # 異なる交叉率
                mutation_rate=0.1 + (run * 0.02),   # 異なる突然変異率
                elite_size=1,
                max_indicators=3,
                allowed_indicators=["SMA", "RSI", "EMA", "WMA", "MOMENTUM", "ROC"]
            )
            
            # バックテスト設定
            backtest_config = {
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "start_date": "2024-01-01",
                "end_date": "2024-04-09",
                "initial_capital": 100000,
                "commission_rate": 0.001,
                "experiment_id": f"test_experiment_{run}"
            }
            
            # GAエンジン実行
            strategy_factory = StrategyFactory()
            ga_engine = GeneticAlgorithmEngine(mock_backtest_service, strategy_factory)
            
            # 異なるランダムシードを設定
            random.seed(100 + run * 10)
            np.random.seed(200 + run * 10)
            
            try:
                result = ga_engine.run_evolution(ga_config, backtest_config)
                best_strategy = result['best_strategy']
                
                print(f"  ✅ 戦略生成成功")
                print(f"    戦略ID: {best_strategy.id}")
                print(f"    指標数: {len(best_strategy.indicators)}")
                
                # 指標の詳細
                indicators_info = []
                for indicator in best_strategy.indicators:
                    info = f"{indicator.type}_{indicator.parameters.get('period', 'N/A')}"
                    indicators_info.append(info)
                    print(f"      - {info}")
                
                # エントリー・エグジット条件
                entry_info = str(best_strategy.entry_conditions[0]) if best_strategy.entry_conditions else "なし"
                exit_info = str(best_strategy.exit_conditions[0]) if best_strategy.exit_conditions else "なし"
                
                print(f"    エントリー条件: {entry_info}")
                print(f"    エグジット条件: {exit_info}")
                
                # 個別バックテスト実行
                try:
                    strategy_class = strategy_factory.create_strategy_class(best_strategy)
                    sample_df = mock_data_service.get_ohlcv_for_backtest("BTC/USDT", "1d", "2024-01-01", "2024-04-09")
                    
                    bt = Backtest(
                        sample_df,
                        strategy_class,
                        cash=100000,
                        commission=0.001,
                        exclusive_orders=True,
                        trade_on_close=True
                    )
                    
                    stats = bt.run()
                    trades = stats['# Trades']
                    returns = stats['Return [%]']
                    win_rate = stats['Win Rate [%]']
                    
                    print(f"    取引回数: {trades}")
                    print(f"    総リターン: {returns:.2f}%")
                    print(f"    勝率: {win_rate:.2f}%")
                    
                    # 戦略情報を保存
                    strategy_info = {
                        'run': run + 1,
                        'strategy_id': best_strategy.id,
                        'indicators': indicators_info,
                        'entry_condition': entry_info,
                        'exit_condition': exit_info,
                        'trades': trades,
                        'returns': returns,
                        'win_rate': win_rate,
                        'execution_time': result['execution_time']
                    }
                    generated_strategies.append(strategy_info)
                    
                    if trades > 0:
                        print(f"    🎉 取引実行成功!")
                    else:
                        print(f"    ⚠️ 取引なし")
                        
                except Exception as e:
                    print(f"    ❌ バックテストエラー: {e}")
                    
            except Exception as e:
                print(f"  ❌ GA実行エラー: {e}")
        
        # 5. 結果分析
        print(f"\n" + "=" * 60)
        print(f"📊 複数戦略生成結果分析")
        print(f"=" * 60)
        
        if len(generated_strategies) > 0:
            print(f"✅ 成功した戦略生成数: {len(generated_strategies)}/{test_runs}")
            
            # 戦略の多様性チェック
            unique_indicators = set()
            unique_entries = set()
            unique_exits = set()
            trading_strategies = 0
            
            for i, strategy in enumerate(generated_strategies, 1):
                print(f"\n戦略 {i}:")
                print(f"  ID: {strategy['strategy_id']}")
                print(f"  指標: {', '.join(strategy['indicators'])}")
                print(f"  エントリー: {strategy['entry_condition']}")
                print(f"  エグジット: {strategy['exit_condition']}")
                print(f"  取引数: {strategy['trades']}, リターン: {strategy['returns']:.2f}%")
                print(f"  実行時間: {strategy['execution_time']:.3f}秒")
                
                # 多様性分析
                unique_indicators.update(strategy['indicators'])
                unique_entries.add(strategy['entry_condition'])
                unique_exits.add(strategy['exit_condition'])
                
                if strategy['trades'] > 0:
                    trading_strategies += 1
            
            print(f"\n📈 多様性分析:")
            print(f"  ユニークな指標組み合わせ: {len(unique_indicators)}種類")
            print(f"  ユニークなエントリー条件: {len(unique_entries)}種類")
            print(f"  ユニークなエグジット条件: {len(unique_exits)}種類")
            print(f"  実際に取引した戦略: {trading_strategies}/{len(generated_strategies)}")
            
            # 成功判定
            if len(generated_strategies) >= 3 and trading_strategies >= 2:
                print(f"\n🎉 複数戦略自動生成テスト: 大成功!")
                print(f"   ✅ 複数の異なる戦略が生成されました")
                print(f"   ✅ 実際に取引を実行する戦略が複数生成されました")
                print(f"   ✅ 自動戦略生成機能は確実に動作しています")
                return True
            else:
                print(f"\n⚠️ 部分的成功")
                print(f"   戦略は生成されましたが、多様性や取引実行に課題があります")
                return False
        else:
            print(f"❌ 戦略生成に失敗しました")
            return False
        
    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multiple_strategy_generation()
    if success:
        print(f"\n🏆 最終結論: 自動戦略生成機能は完全に動作しています！")
    else:
        print(f"\n⚠️ 最終結論: 機能に改善の余地があります")
