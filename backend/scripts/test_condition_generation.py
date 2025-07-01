#!/usr/bin/env python3
"""
条件生成のテストスクリプト

不適切な条件（close > OBV など）が生成される問題を調査します。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.core.services.auto_strategy.models.ga_config import GAConfig, EvolutionConfig, IndicatorConfig, GeneGenerationConfig
from app.core.services.auto_strategy.models.strategy_gene import IndicatorGene
from app.core.services.auto_strategy.utils.operand_grouping import operand_grouping_system

def test_condition_generation():
    """条件生成をテストして問題を特定"""
    
    print("🔍 条件生成テスト開始")
    print("=" * 50)
    
    # GA設定を作成
    ga_config = GAConfig(
        evolution=EvolutionConfig(
            population_size=5,
            generations=2,
            crossover_rate=0.8,
            mutation_rate=0.2,
        ),
        indicators=IndicatorConfig(
            allowed_indicators=["RSI", "SMA", "OBV", "ADX"],
            max_indicators=3,
        ),
        gene_generation=GeneGenerationConfig(
            numeric_threshold_probability=0.8,  # 80%の確率で数値を使用
            min_compatibility_score=0.8,  # 最小互換性スコア
            strict_compatibility_score=0.9,  # 厳密な互換性スコア
        ),
    )
    
    # ジェネレーターを作成
    generator = RandomGeneGenerator(ga_config)
    
    # テスト用の指標リスト
    test_indicators = [
        IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
        IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
        IndicatorGene(type="OBV", parameters={}, enabled=True),
        IndicatorGene(type="ADX", parameters={"period": 14}, enabled=True),
    ]
    
    print("📊 設定値:")
    print(f"  数値閾値使用確率: {ga_config.gene_generation.numeric_threshold_probability:.1%}")
    print(f"  最小互換性スコア: {ga_config.gene_generation.min_compatibility_score}")
    print(f"  厳密互換性スコア: {ga_config.gene_generation.strict_compatibility_score}")
    print()
    
    print("🧪 互換性スコア確認:")
    operands = ["close", "RSI", "SMA", "OBV", "ADX"]
    for i, op1 in enumerate(operands):
        for op2 in operands[i+1:]:
            score = operand_grouping_system.get_compatibility_score(op1, op2)
            status = "✅" if score >= 0.8 else "⚠️" if score >= 0.3 else "❌"
            print(f"  {op1} vs {op2}: {score:.2f} {status}")
    print()
    
    print("🎲 条件生成テスト (50回):")
    print("-" * 30)
    
    scale_mismatches = 0
    numerical_conditions = 0
    total_conditions = 0
    condition_patterns = {}
    
    for i in range(50):
        try:
            condition = generator._generate_single_condition(test_indicators, "entry")
            total_conditions += 1
            
            left = condition.left_operand
            op = condition.operator
            right = condition.right_operand
            
            pattern = f"{left} {op} {type(right).__name__}"
            condition_patterns[pattern] = condition_patterns.get(pattern, 0) + 1
            
            if isinstance(right, (int, float)):
                numerical_conditions += 1
                print(f"  {i+1:2d}. {left} {op} {right} (数値) ✅")
            else:
                # 互換性をチェック
                compatibility = operand_grouping_system.get_compatibility_score(left, right)
                
                if compatibility < 0.3:
                    scale_mismatches += 1
                    print(f"  {i+1:2d}. {left} {op} {right} (互換性: {compatibility:.2f}) ❌ スケール不一致")
                elif compatibility < 0.8:
                    print(f"  {i+1:2d}. {left} {op} {right} (互換性: {compatibility:.2f}) ⚠️ 低い互換性")
                else:
                    print(f"  {i+1:2d}. {left} {op} {right} (互換性: {compatibility:.2f}) ✅ 高い互換性")
                    
        except Exception as e:
            print(f"  {i+1:2d}. エラー: {e}")
    
    print()
    print("📈 結果サマリー:")
    print(f"  総条件数: {total_conditions}")
    print(f"  数値条件: {numerical_conditions} ({numerical_conditions/total_conditions:.1%})")
    print(f"  スケール不一致: {scale_mismatches} ({scale_mismatches/total_conditions:.1%})")
    print()
    
    print("📋 条件パターン:")
    for pattern, count in sorted(condition_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count}回")
    
    # 問題の分析
    print()
    print("🔍 問題分析:")
    if scale_mismatches > 0:
        print(f"  ❌ スケール不一致の条件が {scale_mismatches} 個生成されました")
        print("  → 条件生成ロジックに問題があります")
    else:
        print("  ✅ スケール不一致の条件は生成されませんでした")
    
    expected_numerical_ratio = ga_config.gene_generation.numeric_threshold_probability
    actual_numerical_ratio = numerical_conditions / total_conditions if total_conditions > 0 else 0
    
    if abs(actual_numerical_ratio - expected_numerical_ratio) > 0.2:
        print(f"  ⚠️ 数値条件の割合が期待値と大きく異なります")
        print(f"     期待値: {expected_numerical_ratio:.1%}, 実際: {actual_numerical_ratio:.1%}")
    else:
        print(f"  ✅ 数値条件の割合は期待値に近いです ({actual_numerical_ratio:.1%})")

if __name__ == "__main__":
    test_condition_generation()
