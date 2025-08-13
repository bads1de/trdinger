#!/usr/bin/env python3
"""
包括的な戦略テスト - 様々なテクニカル指標を使用
"""

from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
from app.services.auto_strategy.models.gene_strategy import StrategyGene, IndicatorGene, Condition
from app.services.auto_strategy.models.gene_position_sizing import PositionSizingGene, PositionSizingMethod
import pandas as pd
import numpy as np
import backtesting

print("=== 包括的な戦略テスト ===")

# テストデータ作成（より現実的なデータ）
np.random.seed(42)
n_periods = 200
base_price = 50000
price_data = []
current_price = base_price

for i in range(n_periods):
    # ランダムウォーク + トレンド
    change = np.random.normal(0, 0.02) + 0.0001  # 小さな上昇トレンド
    current_price *= (1 + change)
    price_data.append(current_price)

# OHLCV データ作成
df = pd.DataFrame({
    "Open": [p * np.random.uniform(0.995, 1.005) for p in price_data],
    "High": [p * np.random.uniform(1.001, 1.02) for p in price_data],
    "Low": [p * np.random.uniform(0.98, 0.999) for p in price_data],
    "Close": price_data,
    "Volume": [np.random.uniform(1000, 5000) for _ in range(n_periods)],
})

print(f"テストデータ作成完了: {len(df)}期間")

# 様々なテクニカル指標をテスト
test_indicators = [
    # 基本移動平均
    {"type": "SMA", "parameters": {"period": 20}},
    {"type": "EMA", "parameters": {"period": 14}},
    {"type": "WMA", "parameters": {"period": 10}},
    
    # オシレーター
    {"type": "RSI", "parameters": {"period": 14}},
    {"type": "MACD", "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9}},
    {"type": "STOCH", "parameters": {"k_period": 14, "d_period": 3}},
    
    # トレンド指標
    {"type": "ADX", "parameters": {"period": 14}},
    {"type": "AROON", "parameters": {"period": 14}},
    
    # ボリューム指標
    {"type": "OBV", "parameters": {}},
    
    # ボラティリティ指標
    {"type": "ATR", "parameters": {"period": 14}},
    {"type": "BB", "parameters": {"period": 20, "std_dev": 2.0}},
]

factory = StrategyFactory()
successful_strategies = []
failed_indicators = []

for i, indicator_config in enumerate(test_indicators):
    print(f"\n=== テスト {i+1}: {indicator_config['type']} ===")
    
    try:
        # 指標遺伝子作成
        indicator_gene = IndicatorGene(
            type=indicator_config["type"],
            parameters=indicator_config["parameters"],
            enabled=True
        )
        
        # 簡単な条件作成
        condition = Condition(
            left_operand=indicator_config["type"],
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
        
        print(f"戦略遺伝子作成成功: {indicator_config['type']}")
        
        # 戦略クラス作成
        strategy_class = factory.create_strategy_class(strategy_gene)
        print(f"戦略クラス作成成功: {indicator_config['type']}")
        
        # バックテスト実行
        bt = backtesting.Backtest(df, strategy_class)
        result = bt.run(strategy_gene=strategy_gene)
        
        print(f"✅ バックテスト成功: {indicator_config['type']}")
        print(f"   総リターン: {result.get('Return [%]', 'N/A'):.2f}%")
        print(f"   取引数: {result.get('# Trades', 'N/A')}")
        print(f"   勝率: {result.get('Win Rate [%]', 'N/A'):.2f}%")
        
        # 戦略インスタンス確認
        strategy_instance = bt._strategy
        
        # 指標登録確認
        indicator_registered = hasattr(strategy_instance, indicator_config['type'])
        print(f"   指標登録: {'✅' if indicator_registered else '❌'}")
        
        if indicator_registered:
            indicator_value = getattr(strategy_instance, indicator_config['type'])
            print(f"   指標値タイプ: {type(indicator_value)}")
            if hasattr(indicator_value, '__len__'):
                print(f"   指標値長さ: {len(indicator_value)}")
        
        successful_strategies.append({
            "indicator": indicator_config['type'],
            "result": result,
            "indicator_registered": indicator_registered
        })
        
    except Exception as e:
        print(f"❌ エラー: {indicator_config['type']} - {e}")
        failed_indicators.append({
            "indicator": indicator_config['type'],
            "error": str(e)
        })
        import traceback
        traceback.print_exc()

print(f"\n=== 総合結果 ===")
print(f"成功した戦略: {len(successful_strategies)}/{len(test_indicators)}")
print(f"失敗した指標: {len(failed_indicators)}")

if successful_strategies:
    print(f"\n✅ 成功した指標:")
    for strategy in successful_strategies:
        indicator_status = "指標登録済み" if strategy["indicator_registered"] else "指標未登録"
        print(f"  - {strategy['indicator']}: {indicator_status}")

if failed_indicators:
    print(f"\n❌ 失敗した指標:")
    for failed in failed_indicators:
        print(f"  - {failed['indicator']}: {failed['error']}")

# 指標登録問題の詳細分析
print(f"\n=== 指標登録問題の分析 ===")
registered_count = sum(1 for s in successful_strategies if s["indicator_registered"])
print(f"指標が正しく登録された戦略: {registered_count}/{len(successful_strategies)}")

if registered_count < len(successful_strategies):
    print("⚠️ 一部の指標が戦略インスタンスに登録されていません")
    print("これは条件評価に影響する可能性があります")
else:
    print("✅ すべての指標が正しく登録されています")
