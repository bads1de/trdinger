#!/usr/bin/env python3
"""
指標登録問題の詳細デバッグ
"""

from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
from app.services.auto_strategy.models.gene_strategy import StrategyGene, IndicatorGene, Condition
from app.services.auto_strategy.models.gene_position_sizing import PositionSizingGene, PositionSizingMethod
import pandas as pd
import numpy as np
import backtesting

print("=== 指標登録問題の詳細デバッグ ===")

# 簡単なテストデータ
df = pd.DataFrame({
    "Open": np.linspace(50, 60, 100),
    "High": np.linspace(51, 61, 100),
    "Low": np.linspace(49, 59, 100),
    "Close": np.linspace(50, 60, 100),
    "Volume": np.full(100, 1000),
})

# SMA指標でテスト
sma_gene = IndicatorGene(type="SMA", parameters={"period": 14}, enabled=True)
condition = Condition(left_operand="SMA", operator=">", right_operand="close")
position_sizing_gene = PositionSizingGene(method=PositionSizingMethod.FIXED_RATIO, fixed_ratio=0.1)

strategy_gene = StrategyGene(
    indicators=[sma_gene],
    long_entry_conditions=[condition],
    short_entry_conditions=[],
    exit_conditions=[condition],
    risk_management={"stop_loss": 0.02, "take_profit": 0.04},
    position_sizing_gene=position_sizing_gene,
)

factory = StrategyFactory()
strategy_class = factory.create_strategy_class(strategy_gene)

print("戦略クラス作成成功")

# 戦略クラスの詳細調査
print(f"\n=== 戦略クラス詳細 ===")
print(f"戦略クラス名: {strategy_class.__name__}")
print(f"戦略クラスのMRO: {strategy_class.__mro__}")

# 戦略クラスのメソッド確認
class_methods = [method for method in dir(strategy_class) if not method.startswith('_')]
print(f"戦略クラスのメソッド: {class_methods}")

# init メソッドの詳細確認
if hasattr(strategy_class, 'init'):
    print(f"init メソッド存在: True")
    import inspect
    init_source = inspect.getsource(strategy_class.init)
    print(f"init メソッドのソース:\n{init_source}")
else:
    print(f"init メソッド存在: False")

# バックテスト実行前の戦略インスタンス作成
print(f"\n=== 手動戦略インスタンス作成 ===")
try:
    # 手動でインスタンス作成
    manual_strategy = strategy_class()
    print("手動戦略インスタンス作成成功")
    
    # データ設定（可能であれば）
    try:
        # backtesting.pyのデータ形式をシミュレート
        class FakeData:
            def __init__(self, df):
                self.df = df
                self.Open = df['Open'].values
                self.High = df['High'].values
                self.Low = df['Low'].values
                self.Close = df['Close'].values
                self.Volume = df['Volume'].values
                
        fake_data = FakeData(df)
        
        # 戦略インスタンスにデータを設定（可能であれば）
        if hasattr(manual_strategy, '_data'):
            manual_strategy._data = fake_data
        
        print("データ設定成功")
        
        # init メソッド手動実行
        if hasattr(manual_strategy, 'init'):
            print("init メソッド手動実行開始...")
            manual_strategy.init()
            print("init メソッド手動実行完了")
            
            # 指標確認
            attrs_after_init = [attr for attr in dir(manual_strategy) if not attr.startswith('_')]
            print(f"init後の属性: {attrs_after_init}")
            
            if hasattr(manual_strategy, 'SMA'):
                print(f"✅ SMA登録成功: {type(manual_strategy.SMA)}")
            else:
                print(f"❌ SMA登録失敗")
                
        else:
            print("init メソッドが存在しません")
            
    except Exception as e:
        print(f"データ設定/init実行エラー: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"手動戦略インスタンス作成エラー: {e}")
    import traceback
    traceback.print_exc()

# backtesting.py での実行
print(f"\n=== backtesting.py実行 ===")
try:
    bt = backtesting.Backtest(df, strategy_class, cash=1000000)  # 十分な資金を設定
    
    # バックテスト実行前の状態確認
    print("バックテスト実行前...")
    
    result = bt.run(strategy_gene=strategy_gene)
    print("バックテスト実行成功")
    
    # 戦略インスタンス取得
    strategy_instance = bt._strategy
    print(f"戦略インスタンス取得: {type(strategy_instance)}")
    
    # 詳細な属性確認
    all_attrs = dir(strategy_instance)
    public_attrs = [attr for attr in all_attrs if not attr.startswith('_')]
    private_attrs = [attr for attr in all_attrs if attr.startswith('_') and not attr.startswith('__')]
    
    print(f"パブリック属性: {public_attrs}")
    print(f"プライベート属性: {private_attrs}")
    
    # 戦略遺伝子確認
    if hasattr(strategy_instance, 'strategy_gene'):
        print(f"戦略遺伝子存在: True")
        print(f"指標遺伝子: {[ind.type for ind in strategy_instance.strategy_gene.indicators]}")
    else:
        print(f"戦略遺伝子存在: False")
    
    # SMA確認
    if hasattr(strategy_instance, 'SMA'):
        print(f"✅ SMA登録成功")
        sma_value = strategy_instance.SMA
        print(f"SMA型: {type(sma_value)}")
        if hasattr(sma_value, '__len__'):
            print(f"SMA長さ: {len(sma_value)}")
    else:
        print(f"❌ SMA登録失敗")
        
        # 代替的な確認方法
        print("代替確認方法:")
        for attr in public_attrs:
            attr_value = getattr(strategy_instance, attr)
            if hasattr(attr_value, '__len__') and not isinstance(attr_value, str):
                try:
                    print(f"  {attr}: {type(attr_value)} (長さ: {len(attr_value)})")
                except:
                    print(f"  {attr}: {type(attr_value)}")
    
except Exception as e:
    print(f"バックテスト実行エラー: {e}")
    import traceback
    traceback.print_exc()
