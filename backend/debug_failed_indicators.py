#!/usr/bin/env python3
"""
失敗した8個の指標の詳細デバッグ
"""

from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
from app.services.auto_strategy.models.gene_strategy import StrategyGene, IndicatorGene, Condition
from app.services.auto_strategy.models.gene_position_sizing import PositionSizingGene, PositionSizingMethod
from app.services.indicators.config import indicator_registry
import pandas as pd
import numpy as np
import backtesting

print("=== 失敗した8個の指標の詳細デバッグ ===")

# 失敗した指標リスト
failed_indicators = ["ALMA", "HMA", "HT_TRENDLINE", "PPO", "RMA", "SWMA", "VWAP", "ZLMA"]

# テストデータ作成
np.random.seed(42)
n_periods = 300
base_price = 50000
price_data = []
volume_data = []
current_price = base_price

for i in range(n_periods):
    change = np.random.normal(0, 0.015) + 0.0001
    current_price *= (1 + change)
    price_data.append(current_price)
    volume_data.append(np.random.uniform(1000, 10000))

df = pd.DataFrame({
    "Open": [p * np.random.uniform(0.995, 1.005) for p in price_data],
    "High": [p * np.random.uniform(1.001, 1.03) for p in price_data],
    "Low": [p * np.random.uniform(0.97, 0.999) for p in price_data],
    "Close": price_data,
    "Volume": volume_data,
})

def get_default_parameters(indicator_name):
    config = indicator_registry.get_indicator_config(indicator_name)
    if not config or not config.parameters:
        return {}
    
    params = {}
    for param_name, param_config in config.parameters.items():
        params[param_name] = param_config.default_value
    return params

factory = StrategyFactory()

for indicator_name in failed_indicators:
    print(f"\n=== 詳細デバッグ: {indicator_name} ===")
    
    try:
        # 指標設定確認
        config = indicator_registry.get_indicator_config(indicator_name)
        if config:
            print(f"指標設定: カテゴリ={config.category}, 結果タイプ={config.result_type.value}")
            print(f"必要データ: {config.required_data}")
            print(f"パラメータ: {config.parameters}")
        else:
            print("指標設定なし")
            continue
        
        # デフォルトパラメータ取得
        default_params = get_default_parameters(indicator_name)
        print(f"デフォルトパラメータ: {default_params}")
        
        # 指標遺伝子作成
        indicator_gene = IndicatorGene(
            type=indicator_name,
            parameters=default_params,
            enabled=True
        )
        
        # 条件作成
        if config.result_type.value == "complex":
            condition = Condition(
                left_operand=f"{indicator_name}_0",
                operator=">",
                right_operand="close"
            )
        else:
            condition = Condition(
                left_operand=indicator_name,
                operator=">",
                right_operand="close"
            )
        
        # ポジションサイジング遺伝子
        position_sizing_gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            fixed_ratio=0.1
        )
        
        # 戦略遺伝子作成
        strategy_gene = StrategyGene(
            indicators=[indicator_gene],
            long_entry_conditions=[condition],
            short_entry_conditions=[],
            exit_conditions=[condition],
            risk_management={"stop_loss": 0.02, "take_profit": 0.04},
            position_sizing_gene=position_sizing_gene,
        )
        
        # 戦略クラス作成
        strategy_class = factory.create_strategy_class(strategy_gene)
        print(f"戦略クラス作成成功")
        
        # バックテスト実行
        bt = backtesting.Backtest(df, strategy_class, cash=1000000)
        result = bt.run(strategy_gene=strategy_gene)
        print(f"バックテスト実行成功")
        
        # 戦略インスタンス詳細確認
        strategy_instance = bt._strategy
        
        # 全属性確認
        all_attrs = [attr for attr in dir(strategy_instance) if not attr.startswith('_')]
        print(f"戦略インスタンス属性: {all_attrs}")
        
        # 指標登録確認
        if config.result_type.value == "complex":
            # 複数出力指標の場合
            for i in range(5):  # 最大5個の出力を確認
                attr_name = f"{indicator_name}_{i}"
                if hasattr(strategy_instance, attr_name):
                    print(f"✅ {attr_name} 登録成功")
                else:
                    print(f"❌ {attr_name} 登録失敗")
                    break
        else:
            # 単一出力指標の場合
            if hasattr(strategy_instance, indicator_name):
                print(f"✅ {indicator_name} 登録成功")
                indicator_value = getattr(strategy_instance, indicator_name)
                print(f"指標値タイプ: {type(indicator_value)}")
                if hasattr(indicator_value, '__len__'):
                    print(f"指標値長さ: {len(indicator_value)}")
            else:
                print(f"❌ {indicator_name} 登録失敗")
        
        # indicators辞書確認
        if hasattr(strategy_instance, 'indicators'):
            print(f"indicators辞書: {list(strategy_instance.indicators.keys())}")
        else:
            print("indicators辞書なし")
            
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()

print(f"\n=== デバッグ完了 ===")
