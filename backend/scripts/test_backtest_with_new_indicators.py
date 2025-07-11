#!/usr/bin/env python3
"""
新しい指標を使ったバックテストのテスト

修正された指標がバックテストで正常に動作するかを確認します。
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.core.services.backtest_service import BacktestService


def create_test_data() -> pd.DataFrame:
    """テスト用のOHLCVデータを作成"""
    np.random.seed(42)
    
    n = 200  # より長いデータ
    base_price = 100
    
    # 日付インデックスを作成
    start_date = datetime.now() - timedelta(days=n)
    dates = pd.date_range(start=start_date, periods=n, freq='1H')
    
    # ランダムウォークで価格データを生成
    returns = np.random.normal(0, 0.01, n)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # OHLCV データを生成
    data = {
        'Open': prices * (1 + np.random.normal(0, 0.001, n)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, n),
    }
    
    # 価格の論理的整合性を保証
    for i in range(n):
        data['High'][i] = max(data['Open'][i], data['High'][i], data['Low'][i], data['Close'][i])
        data['Low'][i] = min(data['Open'][i], data['High'][i], data['Low'][i], data['Close'][i])
    
    df = pd.DataFrame(data, index=dates)
    return df


def test_backtest_with_new_indicators():
    """新しい指標を使ったバックテストのテスト"""
    print("=== 新しい指標を使ったバックテストテスト ===")
    
    try:
        # テストデータを作成
        test_data = create_test_data()
        print(f"テストデータ作成完了: {len(test_data)}行")
        
        # 設定を作成
        config = GAConfig(
            population_size=3,
            generations=1,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elite_size=1,
            max_indicators=2,  # 指標数を制限
            min_conditions=1,
            max_conditions=1,  # 条件数を制限
        )
        
        # ランダム遺伝子生成器を作成
        generator = RandomGeneGenerator(config)
        
        # 戦略ファクトリーを作成
        factory = StrategyFactory()
        
        # 複数の戦略をテスト
        for i in range(3):
            print(f"\n戦略 {i+1} をテスト中...")
            
            try:
                # 戦略遺伝子を生成
                gene = generator.generate_random_gene()
                print(f"  遺伝子生成成功")
                
                # 使用されている指標を表示
                indicator_types = [ind.type for ind in gene.indicators if ind.enabled]
                print(f"  使用指標: {indicator_types}")
                
                # 戦略クラスを作成
                strategy_class = factory.create_strategy_class(gene)
                print(f"  戦略クラス作成成功")
                
                # 簡単なバックテストを実行
                import backtesting as bt
                
                bt_result = bt.Backtest(test_data, strategy_class)
                stats = bt_result.run()
                
                print(f"  ✅ バックテスト成功")
                print(f"     総リターン: {stats['Return [%]']:.2f}%")
                print(f"     取引回数: {stats['# Trades']}")
                
            except Exception as e:
                print(f"  ❌ 戦略 {i+1}: エラー - {str(e)}")
        
        print(f"\n=== テスト完了 ===")
        
    except Exception as e:
        print(f"❌ 初期化エラー: {str(e)}")


if __name__ == "__main__":
    test_backtest_with_new_indicators()
