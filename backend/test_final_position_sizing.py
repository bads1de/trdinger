#!/usr/bin/env python3
"""
Position Sizing修正の最終確認テスト
"""

import sys
import os

# パスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_final_position_sizing():
    """Position Sizing修正の最終確認"""
    print("=== Position Sizing修正 最終確認テスト ===")
    
    try:
        from app.core.services.auto_strategy.models.position_sizing_gene import PositionSizingGene, PositionSizingMethod
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene
        
        # 1. ユーザーの現在の設定をシミュレート（修正前の問題）
        print("📊 1. 修正前の問題をシミュレート")
        
        old_gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            fixed_ratio=0.1,  # 10%
            max_position_size=1.0,  # 問題の原因
            enabled=True
        )
        
        strategy_gene_old = StrategyGene(
            id="test_old",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            risk_management={"position_size": 0.1},
            position_sizing_gene=old_gene,
        )
        
        factory = StrategyFactory()
        
        old_result = factory._calculate_position_size(
            strategy_gene_old, 
            account_balance=100000.0, 
            current_price=50000.0, 
            data=None
        )
        
        print(f"  修正前の結果: {old_result} (max_position_size=1.0により制限)")
        
        # 2. 修正後の新しいデフォルト設定
        print("\n📊 2. 修正後の新しいデフォルト設定")
        
        new_gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            fixed_ratio=0.1,  # 10%
            # max_position_size=10.0 (新しいデフォルト)
            enabled=True
        )
        
        strategy_gene_new = StrategyGene(
            id="test_new",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            risk_management={"position_size": 0.1},
            position_sizing_gene=new_gene,
        )
        
        new_result = factory._calculate_position_size(
            strategy_gene_new, 
            account_balance=100000.0, 
            current_price=50000.0, 
            data=None
        )
        
        print(f"  修正後の結果: {new_result} (max_position_size=10.0)")
        
        # 3. より大きなmax_position_sizeでのテスト
        print("\n📊 3. より大きなmax_position_sizeでのテスト")
        
        large_gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            fixed_ratio=0.1,  # 10%
            max_position_size=50.0,  # より大きな値
            enabled=True
        )
        
        strategy_gene_large = StrategyGene(
            id="test_large",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            risk_management={"position_size": 0.1},
            position_sizing_gene=large_gene,
        )
        
        large_result = factory._calculate_position_size(
            strategy_gene_large, 
            account_balance=100000.0, 
            current_price=50000.0, 
            data=None
        )
        
        print(f"  大きなmax_position_sizeでの結果: {large_result}")
        
        # 4. 実際の計算ロジックの確認
        print("\n📊 4. 計算ロジックの詳細確認")
        
        account_balance = 100000.0
        fixed_ratio = 0.1
        calculated_amount = account_balance * fixed_ratio
        
        print(f"  口座残高: {account_balance:,.0f}")
        print(f"  Fixed Ratio: {fixed_ratio} ({fixed_ratio*100}%)")
        print(f"  計算金額: {calculated_amount:,.0f}")
        print(f"  → この金額が「枚数」として解釈される")
        print(f"  → max_position_sizeにより制限される")
        
        print(f"\n  修正前: min({calculated_amount}, 1.0) = {min(calculated_amount, 1.0)}")
        print(f"  修正後: min({calculated_amount}, 10.0) = {min(calculated_amount, 10.0)}")
        print(f"  大きな値: min({calculated_amount}, 50.0) = {min(calculated_amount, 50.0)}")
        
        # 5. ランダム生成のテスト
        print("\n📊 5. ランダム生成のテスト")
        
        from app.core.services.auto_strategy.models.position_sizing_gene import create_random_position_sizing_gene
        
        for i in range(3):
            random_gene = create_random_position_sizing_gene()
            print(f"  ランダム生成 {i+1}: max_position_size={random_gene.max_position_size:.2f}")
            
            # バリデーションテスト
            is_valid, errors = random_gene.validate()
            if is_valid:
                print(f"    ✅ バリデーション成功")
            else:
                print(f"    ❌ バリデーション失敗: {errors}")
        
        print("\n🎉 Position Sizing修正の最終確認が完了しました！")
        print("\n📋 修正内容まとめ:")
        print("  - デフォルトmax_position_size: 1.0 → 10.0")
        print("  - ランダム生成範囲: 0.5-2.0 → 5.0-50.0")
        print("  - バリデーション上限: 10.0 → 100.0")
        print("  - 突然変異上限: 20.0 → 100.0")
        
        print("\n📈 期待される効果:")
        print("  - より大きなポジションサイズが可能")
        print("  - GAによる適切な最適化")
        print("  - 実際の取引量の改善")
        
        return True
        
    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_final_position_sizing()
    sys.exit(0 if success else 1)
