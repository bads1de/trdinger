#!/usr/bin/env python3
"""
Phase 4 新規指標の簡単なテスト
"""

import sys
import os
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_simple_strategy():
    """簡単な戦略テスト"""
    print("🧪 Phase 4 簡単な戦略テスト")
    print("=" * 70)
    
    try:
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene, IndicatorGene, Condition
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
        
        # 簡単な戦略: PLUS_DI + 価格比較
        strategy = StrategyGene(
            id="simple_plus_di",
            indicators=[
                IndicatorGene(
                    type="PLUS_DI",
                    parameters={"period": 14},
                    enabled=True
                )
            ],
            entry_conditions=[
                Condition(
                    left_operand="PLUS_DI_14",
                    operator=">",
                    right_operand=25.0
                )
            ],
            exit_conditions=[
                Condition(
                    left_operand="close",
                    operator="<",
                    right_operand=50000.0
                )
            ],
            risk_management={
                "stop_loss": 0.02,
                "take_profit": 0.03
            }
        )
        
        print("📊 戦略遺伝子作成完了")
        
        # StrategyFactory初期化
        factory = StrategyFactory()
        print("✅ StrategyFactory初期化完了")
        
        # 遺伝子検証
        is_valid, errors = factory.validate_gene(strategy)
        print(f"📊 遺伝子検証結果: {is_valid}")
        if not is_valid:
            print(f"❌ 検証エラー: {errors}")
            
            # 利用可能指標を確認
            available_indicators = []
            for ind in strategy.indicators:
                if ind.enabled:
                    if ind.type == "PLUS_DI":
                        available_indicators.append(f"PLUS_DI_{ind.parameters.get('period', '')}")
                    else:
                        available_indicators.append(f"{ind.type}_{ind.parameters.get('period', '')}")
            
            print(f"📋 利用可能指標: {available_indicators}")
            
            # 条件の詳細確認
            for i, condition in enumerate(strategy.entry_conditions):
                print(f"📋 エントリー条件{i}: {condition.left_operand} {condition.operator} {condition.right_operand}")
                
                # 指標名の検証
                from app.core.services.auto_strategy.models.strategy_gene import Condition as ConditionClass
                test_condition = ConditionClass(condition.left_operand, condition.operator, condition.right_operand)
                is_indicator = test_condition._is_indicator_name(condition.left_operand)
                print(f"  指標名判定: {is_indicator}")
            
            return False
        else:
            print("✅ 遺伝子検証成功")
            return True
        
    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行関数"""
    print("🚀 Phase 4 簡単なテスト開始")
    print("=" * 70)
    
    success = test_simple_strategy()
    
    print("\n" + "=" * 70)
    print("📊 最終結果")
    print("=" * 70)
    
    if success:
        print("🎉 テストが成功しました！")
    else:
        print("❌ テストが失敗しました")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
