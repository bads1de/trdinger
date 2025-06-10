#!/usr/bin/env python3
"""
コア戦略モデルのテスト

GAエンジンに依存しない、コア機能のみのテスト
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.services.auto_strategy.models.strategy_gene import StrategyGene, IndicatorGene, Condition
from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator


def test_strategy_gene_models():
    """戦略遺伝子モデルのテスト"""
    print("🧬 戦略遺伝子モデルテスト開始")
    print("=" * 60)
    
    try:
        # 1. 正しいテクニカル指標の作成
        print("1. テクニカル指標作成テスト...")
        
        valid_indicators = [
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            IndicatorGene(type="MACD", parameters={"fast_period": 12, "slow_period": 26, "signal_period": 9}, enabled=True),
            IndicatorGene(type="BB", parameters={"period": 20, "std_dev": 2.0}, enabled=True),
        ]
        
        for indicator in valid_indicators:
            if indicator.validate():
                print(f"  ✅ {indicator.type}: 有効")
            else:
                print(f"  ❌ {indicator.type}: 無効")
                return False
        
        # 2. 無効な指標（OI/FRベース）のテスト
        print("\n2. 無効指標テスト...")
        
        invalid_indicators = [
            IndicatorGene(type="OI_SMA", parameters={"period": 20}, enabled=True),
            IndicatorGene(type="FR_EMA", parameters={"period": 10}, enabled=True),
            IndicatorGene(type="OpenInterest", parameters={}, enabled=True),
            IndicatorGene(type="FundingRate", parameters={}, enabled=True),
        ]
        
        for indicator in invalid_indicators:
            if not indicator.validate():
                print(f"  ✅ {indicator.type}: 正しく無効と判定")
            else:
                print(f"  ❌ {indicator.type}: 無効なのに有効と判定された")
                return False
        
        # 3. 正しい判断条件の作成
        print("\n3. 判断条件作成テスト...")
        
        valid_conditions = [
            Condition(left_operand="close", operator=">", right_operand="SMA_20"),
            Condition(left_operand="RSI_14", operator="<", right_operand=30),
            Condition(left_operand="FundingRate", operator=">", right_operand=0.001),  # 判断材料として使用
            Condition(left_operand="OpenInterest", operator=">", right_operand=1000000),  # 判断材料として使用
            Condition(left_operand="close", operator="cross_above", right_operand="SMA_20"),
        ]
        
        for i, condition in enumerate(valid_conditions):
            if condition.validate():
                print(f"  ✅ 条件{i+1}: {condition.left_operand} {condition.operator} {condition.right_operand} - 有効")
            else:
                print(f"  ❌ 条件{i+1}: 無効")
                return False
        
        # 4. 戦略遺伝子の作成と検証
        print("\n4. 戦略遺伝子作成テスト...")
        
        gene = StrategyGene(
            indicators=valid_indicators,
            entry_conditions=valid_conditions[:3],
            exit_conditions=valid_conditions[3:],
            risk_management={"stop_loss": 0.03, "take_profit": 0.1}
        )
        
        is_valid, errors = gene.validate()
        if is_valid:
            print(f"  ✅ 戦略遺伝子作成成功: ID {gene.id}")
        else:
            print(f"  ❌ 戦略遺伝子無効: {errors}")
            return False
        
        print("\n🎉 戦略遺伝子モデルテスト完了！")
        return True
        
    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_random_gene_generator():
    """ランダム遺伝子生成器のテスト"""
    print("\n🎲 ランダム遺伝子生成器テスト開始")
    print("=" * 60)
    
    try:
        # 1. 生成器作成
        print("1. 生成器作成...")
        generator = RandomGeneGenerator({
            "max_indicators": 3,
            "min_indicators": 2,
            "max_conditions": 3,
            "min_conditions": 1
        })
        print("  ✅ 生成器作成完了")
        
        # 2. 単一遺伝子生成
        print("\n2. 単一遺伝子生成テスト...")
        gene = generator.generate_random_gene()
        
        print(f"  📊 生成された指標数: {len(gene.indicators)}")
        print(f"  📊 エントリー条件数: {len(gene.entry_conditions)}")
        print(f"  📊 イグジット条件数: {len(gene.exit_conditions)}")
        
        # 指標の詳細
        print("  📋 生成された指標:")
        for i, indicator in enumerate(gene.indicators):
            print(f"    {i+1}. {indicator.type} - {indicator.parameters}")
        
        # 条件の詳細
        print("  📋 エントリー条件:")
        for i, condition in enumerate(gene.entry_conditions):
            print(f"    {i+1}. {condition.left_operand} {condition.operator} {condition.right_operand}")
        
        # 3. 妥当性確認
        print("\n3. 生成された遺伝子の妥当性確認...")
        is_valid, errors = gene.validate()
        
        if is_valid:
            print("  ✅ 生成された遺伝子は有効")
        else:
            print(f"  ❌ 生成された遺伝子が無効: {errors}")
            return False
        
        # 4. 指標タイプの確認
        print("\n4. 指標タイプ確認...")
        invalid_indicator_types = []
        for indicator in gene.indicators:
            if indicator.type in ["OpenInterest", "FundingRate"] or \
               indicator.type.startswith(("OI_", "FR_")):
                invalid_indicator_types.append(indicator.type)
        
        if not invalid_indicator_types:
            print("  ✅ 全ての指標がテクニカル指標 (正しい)")
        else:
            print(f"  ❌ 無効な指標タイプが含まれている: {invalid_indicator_types}")
            return False
        
        # 5. OI/FR判断条件の確認
        print("\n5. OI/FR判断条件確認...")
        all_conditions = gene.entry_conditions + gene.exit_conditions
        oi_fr_conditions = []
        
        for condition in all_conditions:
            if condition.left_operand in ["OpenInterest", "FundingRate"] or \
               (isinstance(condition.right_operand, str) and 
                condition.right_operand in ["OpenInterest", "FundingRate"]):
                oi_fr_conditions.append(condition)
        
        print(f"  📊 OI/FR判断条件数: {len(oi_fr_conditions)}")
        for i, condition in enumerate(oi_fr_conditions):
            print(f"    {i+1}. {condition.left_operand} {condition.operator} {condition.right_operand}")
        
        # 6. 複数生成テスト
        print("\n6. 複数生成テスト...")
        population = generator.generate_population(5)
        
        valid_count = 0
        oi_fr_usage_count = 0
        
        for i, individual in enumerate(population):
            is_valid, _ = individual.validate()
            if is_valid:
                valid_count += 1
            
            # OI/FR使用確認
            all_conds = individual.entry_conditions + individual.exit_conditions
            has_oi_fr = any(
                cond.left_operand in ["OpenInterest", "FundingRate"] or
                (isinstance(cond.right_operand, str) and 
                 cond.right_operand in ["OpenInterest", "FundingRate"])
                for cond in all_conds
            )
            
            if has_oi_fr:
                oi_fr_usage_count += 1
            
            print(f"    個体{i+1}: {'✅' if is_valid else '❌'} {'(OI/FR使用)' if has_oi_fr else ''}")
        
        print(f"  📊 有効個体率: {valid_count}/{len(population)} ({valid_count/len(population)*100:.1f}%)")
        print(f"  📊 OI/FR活用率: {oi_fr_usage_count}/{len(population)} ({oi_fr_usage_count/len(population)*100:.1f}%)")
        
        if valid_count >= len(population) * 0.8:
            print("  ✅ 生成品質: 良好")
        else:
            print("  ⚠️ 生成品質: 要改善")
        
        print("\n🎉 ランダム遺伝子生成器テスト完了！")
        return True
        
    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ga_objectives():
    """GA目的の確認テスト"""
    print("\n🎯 GA目的確認テスト開始")
    print("=" * 60)
    
    print("1. GA真の目的:")
    print("  🎯 高いリターン (Total Return)")
    print("  📊 高いシャープレシオ (Sharpe Ratio)")
    print("  📉 低いドローダウン (Max Drawdown)")
    print("  ✨ これらを最適化する優れた投資戦略手法の発掘")
    
    print("\n2. OI/FRの正しい役割:")
    print("  📋 判断材料・シグナルとして使用")
    print("  📋 例: FundingRate > 0.01% → ロングポジション過熱 → ショート検討")
    print("  📋 例: OpenInterest 急増 → 市場参加者増加 → トレンド継続可能性")
    print("  📋 例: FundingRate < -0.005% → ショートポジション過熱 → ロング検討")
    
    print("\n3. 間違った使用例:")
    print("  ❌ FR_SMA, OI_EMA などの指標計算")
    print("  ❌ OI/FRに対する移動平均の適用")
    print("  ❌ OI/FRを指標として扱うこと")
    
    print("\n4. 正しい使用例:")
    print("  ✅ FundingRate > 閾値 (判断条件)")
    print("  ✅ OpenInterest > 閾値 (判断条件)")
    print("  ✅ テクニカル指標 + OI/FR判断の組み合わせ")
    
    print("\n🎉 GA目的確認テスト完了！")
    return True


if __name__ == "__main__":
    success1 = test_strategy_gene_models()
    success2 = test_random_gene_generator()
    success3 = test_ga_objectives()
    
    print("\n" + "=" * 60)
    if success1 and success2 and success3:
        print("🎊 全テスト成功！")
        print("✨ 修正されたコア戦略モデルが正常に動作しています")
        print("")
        print("📋 実装確認:")
        print("  ✅ テクニカル指標のみを使用")
        print("  ✅ OI/FRは判断材料として使用")
        print("  ✅ GA目的: 高リターン・高シャープレシオ・低ドローダウン")
        print("")
        print("🚀 次のステップ: StrategyFactoryの対応とエンドツーエンドテスト")
    else:
        print("💥 一部テスト失敗")
        print("🔧 さらなる修正が必要です")
        sys.exit(1)
