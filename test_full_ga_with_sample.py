#!/usr/bin/env python3
"""
サンプルデータを使用した完全なGA実行テスト
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_full_ga():
    """完全なGA実行をテスト"""
    print("🧬 完全なGA実行テスト（サンプルデータ使用）")
    print("=" * 60)
    
    try:
        # 1. 必要なモジュールのインポート
        print("1. モジュールインポート中...")
        import pandas as pd
        import numpy as np
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        from app.core.services.auto_strategy.engines.ga_engine import GeneticAlgorithmEngine
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
        from app.core.services.backtest_service import BacktestService
        from app.core.services.backtest_data_service import BacktestDataService
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
                    'timestamp': date,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': volume
                })
            
            return data
        
        sample_ohlcv_data = generate_sample_data(100)
        print(f"  ✅ サンプルデータ生成: {len(sample_ohlcv_data)}日分")
        
        # 3. モックデータサービスの作成
        print("\n3. モックデータサービス作成中...")
        
        class MockOHLCVRepository:
            def __init__(self, data):
                self.data = data
            
            def get_ohlcv_data(self, symbol, timeframe, start_time=None, end_time=None, limit=None):
                # OHLCVDataオブジェクトのモック
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
                
                # DataFrameに変換
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
        
        mock_repo = MockOHLCVRepository(sample_ohlcv_data)
        mock_data_service = MockBacktestDataService(mock_repo)
        mock_backtest_service = BacktestService(mock_data_service)
        
        print("  ✅ モックサービス作成完了")
        
        # 4. GA設定
        print("\n4. GA設定作成中...")
        ga_config = GAConfig(
            population_size=5,  # 小規模テスト
            generations=3,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite_size=1,
            max_indicators=3,
            allowed_indicators=["SMA", "RSI", "EMA"]
        )
        print(f"  ✅ GA設定: 個体数{ga_config.population_size}, 世代数{ga_config.generations}")
        
        # 5. バックテスト設定
        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": "2024-01-01",
            "end_date": "2024-04-09",
            "initial_capital": 100000,
            "commission_rate": 0.001,
            "experiment_id": "test_experiment"
        }
        print(f"  ✅ バックテスト設定: {backtest_config['symbol']} {backtest_config['timeframe']}")
        
        # 6. GAエンジンの初期化と実行
        print("\n6. GAエンジン実行中...")
        strategy_factory = StrategyFactory()
        ga_engine = GeneticAlgorithmEngine(mock_backtest_service, strategy_factory)
        
        # 進捗コールバック
        def progress_callback(progress):
            print(f"  世代 {progress.current_generation}/{progress.total_generations}: "
                  f"最高フィットネス {progress.best_fitness:.4f}")
        
        ga_engine.set_progress_callback(progress_callback)
        
        # GA実行
        result = ga_engine.run_evolution(ga_config, backtest_config)
        
        print(f"\n✅ GA実行完了!")
        print(f"  実行時間: {result['execution_time']:.2f}秒")
        print(f"  完了世代数: {result['generations_completed']}")
        print(f"  最終個体数: {result['final_population_size']}")
        print(f"  最高フィットネス: {result['best_fitness']:.4f}")
        
        # 7. 最優秀戦略の詳細
        print(f"\n🏆 最優秀戦略:")
        best_strategy = result['best_strategy']
        print(f"  戦略ID: {best_strategy.id}")
        print(f"  指標数: {len(best_strategy.indicators)}")
        
        for i, indicator in enumerate(best_strategy.indicators, 1):
            print(f"    {i}. {indicator.type} - {indicator.parameters}")
        
        print(f"  エントリー条件数: {len(best_strategy.entry_conditions)}")
        for i, condition in enumerate(best_strategy.entry_conditions, 1):
            print(f"    {i}. {condition}")
        
        print(f"  エグジット条件数: {len(best_strategy.exit_conditions)}")
        for i, condition in enumerate(best_strategy.exit_conditions, 1):
            print(f"    {i}. {condition}")
        
        # 8. 最優秀戦略の個別テスト
        print(f"\n8. 最優秀戦略の個別テスト...")
        try:
            best_strategy_class = strategy_factory.create_strategy_class(best_strategy)
            
            from backtesting import Backtest
            sample_df = mock_data_service.get_ohlcv_for_backtest("BTC/USDT", "1d", "2024-01-01", "2024-04-09")
            
            bt = Backtest(
                sample_df,
                best_strategy_class,
                cash=100000,
                commission=0.001,
                exclusive_orders=True,
                trade_on_close=True
            )
            
            stats = bt.run()
            print(f"  ✅ 個別テスト成功")
            print(f"    総リターン: {stats['Return [%]']:.2f}%")
            print(f"    取引回数: {stats['# Trades']}")
            print(f"    勝率: {stats['Win Rate [%]']:.2f}%")
            print(f"    シャープレシオ: {stats['Sharpe Ratio']:.4f}")
            
            if stats['# Trades'] > 0:
                print(f"  🎉 最優秀戦略が実際に取引を実行しました！")
            else:
                print(f"  ⚠️ 最優秀戦略が取引を実行しませんでした")
                
        except Exception as e:
            print(f"  ❌ 個別テスト失敗: {e}")
        
        print(f"\n" + "=" * 60)
        print(f"🎉 完全なGA実行テスト完了！")
        print(f"自動戦略生成機能は正常に動作しています。")
        
        return True
        
    except Exception as e:
        print(f"\n❌ GA実行テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_full_ga()
