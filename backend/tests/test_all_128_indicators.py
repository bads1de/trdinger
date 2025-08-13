#!/usr/bin/env python3
"""
128個すべてのテクニカル指標をオートストラテジーでテスト
"""

from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
from app.services.auto_strategy.models.gene_strategy import StrategyGene, IndicatorGene, Condition
from app.services.auto_strategy.models.gene_position_sizing import PositionSizingGene, PositionSizingMethod
from app.services.indicators.config import indicator_registry
import pandas as pd
import numpy as np
import backtesting
import traceback

print("=== 128個すべてのテクニカル指標テスト ===")

# より現実的なテストデータ作成
np.random.seed(42)
n_periods = 300  # より多くのデータ
base_price = 50000
price_data = []
volume_data = []
current_price = base_price

for i in range(n_periods):
    # ランダムウォーク + トレンド + ボラティリティ
    change = np.random.normal(0, 0.015) + 0.0001  # 小さな上昇トレンド
    current_price *= (1 + change)
    price_data.append(current_price)
    volume_data.append(np.random.uniform(1000, 10000))

# OHLCV データ作成
df = pd.DataFrame({
    "Open": [p * np.random.uniform(0.995, 1.005) for p in price_data],
    "High": [p * np.random.uniform(1.001, 1.03) for p in price_data],
    "Low": [p * np.random.uniform(0.97, 0.999) for p in price_data],
    "Close": price_data,
    "Volume": volume_data,
})

print(f"テストデータ作成完了: {len(df)}期間")

# 登録されているすべての指標を取得
all_indicators = indicator_registry.get_supported_indicator_names()
print(f"テスト対象指標数: {len(all_indicators)}")

# 各指標のデフォルトパラメータを取得する関数
def get_default_parameters(indicator_name):
    config = indicator_registry.get_indicator_config(indicator_name)
    if not config or not config.parameters:
        return {}
    
    params = {}
    for param_name, param_config in config.parameters.items():
        params[param_name] = param_config.default_value
    return params

# テスト結果を格納
results = {
    "success": [],
    "failed": [],
    "calculation_failed": [],
    "strategy_creation_failed": [],
    "backtest_failed": [],
    "indicator_registration_failed": []
}

factory = StrategyFactory()

# 各指標をテスト
for i, indicator_name in enumerate(sorted(all_indicators)):
    print(f"\n=== テスト {i+1}/{len(all_indicators)}: {indicator_name} ===")
    
    try:
        # デフォルトパラメータ取得
        default_params = get_default_parameters(indicator_name)
        print(f"デフォルトパラメータ: {default_params}")
        
        # 指標遺伝子作成
        indicator_gene = IndicatorGene(
            type=indicator_name,
            parameters=default_params,
            enabled=True
        )
        
        # 簡単な条件作成（単一出力指標の場合）
        config = indicator_registry.get_indicator_config(indicator_name)
        if config and config.result_type.value == "complex":
            # 複数出力指標の場合、最初の出力を使用
            condition = Condition(
                left_operand=f"{indicator_name}_0",
                operator=">",
                right_operand="close"
            )
        else:
            # 単一出力指標の場合
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
        
        print(f"戦略遺伝子作成成功")
        
        # 戦略クラス作成
        try:
            strategy_class = factory.create_strategy_class(strategy_gene)
            print(f"戦略クラス作成成功")
        except Exception as e:
            print(f"❌ 戦略クラス作成失敗: {e}")
            results["strategy_creation_failed"].append({
                "indicator": indicator_name,
                "error": str(e)
            })
            continue
        
        # バックテスト実行
        try:
            bt = backtesting.Backtest(df, strategy_class, cash=1000000)
            result = bt.run(strategy_gene=strategy_gene)
            print(f"バックテスト実行成功")
            
            # 戦略インスタンス確認
            strategy_instance = bt._strategy
            
            # 指標登録確認
            indicator_registered = False
            if config and config.result_type.value == "complex":
                # 複数出力指標の場合
                indicator_registered = hasattr(strategy_instance, f"{indicator_name}_0")
            else:
                # 単一出力指標の場合
                indicator_registered = hasattr(strategy_instance, indicator_name)
            
            if indicator_registered:
                print(f"✅ 指標登録成功")
                results["success"].append({
                    "indicator": indicator_name,
                    "result": result,
                    "indicator_registered": True
                })
            else:
                print(f"⚠️ 指標登録失敗")
                results["indicator_registration_failed"].append({
                    "indicator": indicator_name,
                    "result": result
                })
                
        except Exception as e:
            print(f"❌ バックテスト実行失敗: {e}")
            results["backtest_failed"].append({
                "indicator": indicator_name,
                "error": str(e)
            })
            continue
            
    except Exception as e:
        print(f"❌ 全体的なエラー: {e}")
        results["failed"].append({
            "indicator": indicator_name,
            "error": str(e)
        })
        # traceback.print_exc()

# 結果サマリー
print(f"\n=== 総合結果 ===")
print(f"完全成功: {len(results['success'])}")
print(f"指標登録失敗: {len(results['indicator_registration_failed'])}")
print(f"戦略作成失敗: {len(results['strategy_creation_failed'])}")
print(f"バックテスト失敗: {len(results['backtest_failed'])}")
print(f"その他失敗: {len(results['failed'])}")

total_tested = len(results['success']) + len(results['indicator_registration_failed']) + len(results['strategy_creation_failed']) + len(results['backtest_failed']) + len(results['failed'])
print(f"テスト総数: {total_tested}/{len(all_indicators)}")

if results['success']:
    print(f"\n✅ 完全成功した指標 ({len(results['success'])}個):")
    for item in results['success']:
        print(f"  - {item['indicator']}")

if results['indicator_registration_failed']:
    print(f"\n⚠️ 指標登録失敗 ({len(results['indicator_registration_failed'])}個):")
    for item in results['indicator_registration_failed']:
        print(f"  - {item['indicator']}")

if results['strategy_creation_failed']:
    print(f"\n❌ 戦略作成失敗 ({len(results['strategy_creation_failed'])}個):")
    for item in results['strategy_creation_failed']:
        print(f"  - {item['indicator']}: {item['error']}")

if results['backtest_failed']:
    print(f"\n❌ バックテスト失敗 ({len(results['backtest_failed'])}個):")
    for item in results['backtest_failed']:
        print(f"  - {item['indicator']}: {item['error']}")

if results['failed']:
    print(f"\n❌ その他失敗 ({len(results['failed'])}個):")
    for item in results['failed']:
        print(f"  - {item['indicator']}: {item['error']}")

success_rate = len(results['success']) / len(all_indicators) * 100
print(f"\n成功率: {success_rate:.1f}%")
