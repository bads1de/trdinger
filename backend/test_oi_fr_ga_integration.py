"""
OI/FR対応GA統合テスト

OI/FR条件を含む戦略がGAで正しく生成・評価されるかをテストします。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.services.auto_strategy.models.strategy_gene import (
    StrategyGene, IndicatorGene, Condition
)
from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory

def test_random_gene_generation_with_oi_fr():
    """OI/FR条件を含むランダム遺伝子生成テスト"""
    print("=== OI/FR条件ランダム生成テスト ===")
    
    generator = RandomGeneGenerator({
        "max_indicators": 3,
        "min_indicators": 1,
        "max_conditions": 3,
        "min_conditions": 1
    })
    
    oi_fr_genes = []
    total_genes = 20
    
    for i in range(total_genes):
        gene = generator.generate_random_gene()
        
        # OI/FR条件が含まれているかチェック
        all_conditions = gene.entry_conditions + gene.exit_conditions
        has_oi_fr = any(
            condition.left_operand in ["OpenInterest", "FundingRate"] or
            (isinstance(condition.right_operand, str) and 
             condition.right_operand in ["OpenInterest", "FundingRate"])
            for condition in all_conditions
        )
        
        if has_oi_fr:
            oi_fr_genes.append(gene)
    
    print(f"生成された遺伝子数: {total_genes}")
    print(f"OI/FR条件を含む遺伝子数: {len(oi_fr_genes)}")
    print(f"OI/FR条件含有率: {len(oi_fr_genes)/total_genes*100:.1f}%")
    
    # OI/FR条件の詳細表示
    if oi_fr_genes:
        print("\n📋 OI/FR条件の例:")
        for i, gene in enumerate(oi_fr_genes[:3]):  # 最初の3つを表示
            print(f"  遺伝子{i+1} (ID: {gene.id}):")
            for j, condition in enumerate(gene.entry_conditions + gene.exit_conditions):
                if (condition.left_operand in ["OpenInterest", "FundingRate"] or
                    (isinstance(condition.right_operand, str) and 
                     condition.right_operand in ["OpenInterest", "FundingRate"])):
                    print(f"    - {condition.left_operand} {condition.operator} {condition.right_operand}")
    
    return len(oi_fr_genes) > 0

def test_oi_fr_strategy_validation():
    """OI/FR戦略の妥当性検証テスト"""
    print("\n=== OI/FR戦略妥当性検証テスト ===")
    
    factory = StrategyFactory()
    generator = RandomGeneGenerator()
    
    valid_count = 0
    total_count = 10
    
    for i in range(total_count):
        # OI/FR条件を強制的に含む遺伝子を生成
        gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            ],
            entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="SMA_20"),
                Condition(left_operand="FundingRate", operator=">", right_operand=0.0005),
            ],
            exit_conditions=[
                Condition(left_operand="RSI_14", operator=">", right_operand=70),
                Condition(left_operand="OpenInterest", operator="<", right_operand=5000000),
            ]
        )
        
        # 妥当性検証
        is_valid, errors = factory.validate_gene(gene)
        if is_valid:
            valid_count += 1
        else:
            print(f"  ❌ 遺伝子{i+1}が無効: {errors}")
    
    print(f"妥当な遺伝子数: {valid_count}/{total_count}")
    print(f"妥当性率: {valid_count/total_count*100:.1f}%")
    
    return valid_count == total_count

def test_oi_fr_strategy_class_creation():
    """OI/FR戦略クラス作成テスト"""
    print("\n=== OI/FR戦略クラス作成テスト ===")
    
    factory = StrategyFactory()
    
    # 複雑なOI/FR条件を含む戦略
    gene = StrategyGene(
        indicators=[
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            IndicatorGene(type="EMA", parameters={"period": 10}, enabled=True),
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
        ],
        entry_conditions=[
            Condition(left_operand="close", operator=">", right_operand="SMA_20"),
            Condition(left_operand="EMA_10", operator=">", right_operand="SMA_20"),
            Condition(left_operand="FundingRate", operator=">", right_operand=0.001),
            Condition(left_operand="OpenInterest", operator=">", right_operand=10000000),
        ],
        exit_conditions=[
            Condition(left_operand="RSI_14", operator=">", right_operand=75),
            Condition(left_operand="FundingRate", operator="<", right_operand=-0.0005),
        ]
    )
    
    try:
        strategy_class = factory.create_strategy_class(gene)
        print(f"✅ 複雑なOI/FR戦略クラス作成成功: {strategy_class.__name__}")
        
        # 戦略の詳細情報
        print(f"  指標数: {len(gene.indicators)}")
        print(f"  エントリー条件数: {len(gene.entry_conditions)}")
        print(f"  イグジット条件数: {len(gene.exit_conditions)}")
        
        # OI/FR条件の数をカウント
        all_conditions = gene.entry_conditions + gene.exit_conditions
        oi_fr_count = sum(1 for condition in all_conditions 
                         if condition.left_operand in ["OpenInterest", "FundingRate"] or
                            (isinstance(condition.right_operand, str) and 
                             condition.right_operand in ["OpenInterest", "FundingRate"]))
        print(f"  OI/FR条件数: {oi_fr_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ 戦略クラス作成失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_oi_fr_condition_patterns():
    """OI/FR条件パターンテスト"""
    print("\n=== OI/FR条件パターンテスト ===")
    
    factory = StrategyFactory()
    
    # 様々なOI/FR条件パターンをテスト
    test_patterns = [
        # FundingRate条件
        Condition(left_operand="FundingRate", operator=">", right_operand=0.0005),
        Condition(left_operand="FundingRate", operator="<", right_operand=-0.0005),
        Condition(left_operand="FundingRate", operator=">=", right_operand=0.001),
        
        # OpenInterest条件
        Condition(left_operand="OpenInterest", operator=">", right_operand=10000000),
        Condition(left_operand="OpenInterest", operator="<", right_operand=5000000),
        Condition(left_operand="OpenInterest", operator=">=", right_operand=15000000),
        
        # 混合条件
        Condition(left_operand="close", operator=">", right_operand="FundingRate"),
        Condition(left_operand="volume", operator="<", right_operand="OpenInterest"),
    ]
    
    success_count = 0
    
    for i, condition in enumerate(test_patterns):
        try:
            gene = StrategyGene(
                indicators=[
                    IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
                ],
                entry_conditions=[condition],
                exit_conditions=[
                    Condition(left_operand="close", operator="<", right_operand="SMA_20"),
                ]
            )
            
            is_valid, errors = factory.validate_gene(gene)
            if is_valid:
                strategy_class = factory.create_strategy_class(gene)
                print(f"  ✅ パターン{i+1}: {condition.left_operand} {condition.operator} {condition.right_operand}")
                success_count += 1
            else:
                print(f"  ❌ パターン{i+1}が無効: {errors}")
                
        except Exception as e:
            print(f"  💥 パターン{i+1}でエラー: {e}")
    
    print(f"\n成功パターン数: {success_count}/{len(test_patterns)}")
    return success_count > 0

def main():
    """メインテスト実行"""
    print("🧪 OI/FR対応GA統合テスト開始\n")
    
    results = []
    
    # テスト実行
    results.append(test_random_gene_generation_with_oi_fr())
    results.append(test_oi_fr_strategy_validation())
    results.append(test_oi_fr_strategy_class_creation())
    results.append(test_oi_fr_condition_patterns())
    
    # 結果サマリー
    print(f"\n📊 テスト結果サマリー:")
    print(f"  成功: {sum(results)}/{len(results)}")
    print(f"  失敗: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 全てのOI/FR GA統合テストが成功しました！")
        print("✅ OI/FR条件を含む戦略の生成・評価が正常に動作しています。")
    else:
        print("⚠️ 一部のテストが失敗しました。")
    
    return all(results)

if __name__ == "__main__":
    main()
