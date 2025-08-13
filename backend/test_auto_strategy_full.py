#!/usr/bin/env python3
"""
完全なオートストラテジーテスト
"""

from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from app.services.auto_strategy.models.ga_config import GAConfig
import pandas as pd
import numpy as np

print("=== 完全なオートストラテジーテスト ===")

# テストデータ作成
data = type("Data", (), {})()
df = pd.DataFrame({
    "Open": np.linspace(50, 60, 100),
    "High": np.linspace(51, 61, 100),
    "Low": np.linspace(49, 59, 100),
    "Close": np.linspace(50, 60, 100),
    "Volume": np.full(100, 1000),
})
data.df = df

# オートストラテジーサービス作成
service = AutoStrategyService()

# GA設定作成
config = GAConfig.create_fast()
config.indicator_mode = "technical_only"
config.population_size = 2  # 小さなサイズでテスト
config.generations = 1  # 1世代のみ

print(f"GA設定: {config}")

try:
    # オートストラテジー生成
    print("オートストラテジー生成開始...")
    strategies = service.generate_strategies(
        symbol="BTC",
        timeframe="1h", 
        config=config,
        count=1
    )
    
    print(f"✅ 戦略生成成功: {len(strategies)}個")
    
    for i, strategy in enumerate(strategies):
        print(f"\n=== 戦略 {i+1} ===")
        print(f"指標: {[ind.type for ind in strategy.indicators]}")
        print(f"ロング条件: {len(strategy.long_entry_conditions)}")
        print(f"ショート条件: {len(strategy.short_entry_conditions)}")
        print(f"イグジット条件: {len(strategy.exit_conditions)}")
        
        # 戦略ファクトリーでクラス作成
        from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
        factory = StrategyFactory()
        
        try:
            strategy_class = factory.create_strategy_class(strategy)
            print("✅ 戦略クラス作成成功")
            
            # backtesting.pyでテスト
            import backtesting
            bt = backtesting.Backtest(data.df, strategy_class)
            
            result = bt.run(strategy_gene=strategy)
            print("✅ バックテスト実行成功")
            
            # 戦略インスタンス確認
            strategy_instance = bt._strategy
            attrs = [attr for attr in dir(strategy_instance) if not attr.startswith("_")]
            print(f"戦略インスタンス属性: {attrs}")
            
            # 指標確認
            for ind in strategy.indicators:
                if hasattr(strategy_instance, ind.type):
                    print(f"✅ {ind.type}指標登録成功")
                else:
                    print(f"❌ {ind.type}指標登録失敗")
                    
        except Exception as e:
            print(f"戦略テストエラー: {e}")
            import traceback
            traceback.print_exc()
            
except Exception as e:
    print(f"オートストラテジー生成エラー: {e}")
    import traceback
    traceback.print_exc()
