#!/usr/bin/env python3
"""
オートストラテジー生成のテストスクリプト

修正後の戦略生成機能をテストし、多様性を確認します。
"""

import sys
import os
import logging
from typing import List, Dict, Any

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.core.services.auto_strategy.models.gene_encoding import GeneEncoder
from app.core.services.auto_strategy.models.strategy_gene import StrategyGene
from app.core.services.auto_strategy.engines.deap_configurator import DEAPConfigurator
from app.core.services.auto_strategy.models.ga_config import GAConfig

# ログ設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_random_gene_generation(num_tests: int = 10) -> List[StrategyGene]:
    """ランダム戦略遺伝子生成のテスト"""
    logger.info(f"ランダム戦略遺伝子生成テスト開始: {num_tests}回")
    
    generator = RandomGeneGenerator()
    genes = []
    
    for i in range(num_tests):
        try:
            logger.info(f"テスト {i+1}/{num_tests}")
            gene = generator.generate_random_gene()
            genes.append(gene)
            
            # 生成された戦略の詳細を表示
            print(f"\n=== 戦略 {i+1} ===")
            print(f"指標数: {len(gene.indicators)}")
            for j, indicator in enumerate(gene.indicators):
                print(f"  指標{j+1}: {indicator.type} - {indicator.parameters}")
            
            print(f"エントリー条件数: {len(gene.entry_conditions)}")
            for j, condition in enumerate(gene.entry_conditions):
                print(f"  条件{j+1}: {condition.left_operand} {condition.operator} {condition.right_operand}")
                
            print(f"エグジット条件数: {len(gene.exit_conditions)}")
            for j, condition in enumerate(gene.exit_conditions):
                print(f"  条件{j+1}: {condition.left_operand} {condition.operator} {condition.right_operand}")
                
            print(f"メタデータ: {gene.metadata}")
            
        except Exception as e:
            logger.error(f"テスト {i+1} でエラー: {e}")
    
    return genes

def test_fallback_generation(num_tests: int = 5) -> List[StrategyGene]:
    """フォールバック戦略生成のテスト"""
    logger.info(f"フォールバック戦略生成テスト開始: {num_tests}回")
    
    config = GAConfig.create_fast()
    configurator = DEAPConfigurator(RandomGeneGenerator())
    encoder = GeneEncoder()
    
    genes = []
    
    for i in range(num_tests):
        try:
            logger.info(f"フォールバックテスト {i+1}/{num_tests}")
            
            # フォールバック個体を生成
            fallback_individual = configurator._create_fallback_individual(config)
            
            # デコードして戦略遺伝子に変換
            gene = encoder.decode_list_to_strategy_gene(fallback_individual, StrategyGene)
            genes.append(gene)
            
            # 生成された戦略の詳細を表示
            print(f"\n=== フォールバック戦略 {i+1} ===")
            print(f"指標数: {len(gene.indicators)}")
            for j, indicator in enumerate(gene.indicators):
                print(f"  指標{j+1}: {indicator.type} - {indicator.parameters}")
            
            print(f"エントリー条件数: {len(gene.entry_conditions)}")
            for j, condition in enumerate(gene.entry_conditions):
                print(f"  条件{j+1}: {condition.left_operand} {condition.operator} {condition.right_operand}")
                
            print(f"メタデータ: {gene.metadata}")
            
        except Exception as e:
            logger.error(f"フォールバックテスト {i+1} でエラー: {e}")
    
    return genes

def analyze_diversity(genes: List[StrategyGene]) -> Dict[str, Any]:
    """生成された戦略の多様性を分析"""
    logger.info("戦略多様性分析開始")
    
    if not genes:
        return {"error": "分析する戦略がありません"}
    
    # 指標タイプの分布
    indicator_types = {}
    total_indicators = 0
    
    # 条件の分布
    operators = {}
    operands = {}
    
    for gene in genes:
        # 指標分析
        for indicator in gene.indicators:
            indicator_type = indicator.type
            indicator_types[indicator_type] = indicator_types.get(indicator_type, 0) + 1
            total_indicators += 1
        
        # 条件分析
        all_conditions = gene.entry_conditions + gene.exit_conditions
        for condition in all_conditions:
            operator = condition.operator
            operators[operator] = operators.get(operator, 0) + 1
            
            left_operand = condition.left_operand
            operands[left_operand] = operands.get(left_operand, 0) + 1
            
            right_operand = condition.right_operand
            operands[right_operand] = operands.get(right_operand, 0) + 1
    
    analysis = {
        "total_strategies": len(genes),
        "total_indicators": total_indicators,
        "avg_indicators_per_strategy": total_indicators / len(genes) if genes else 0,
        "indicator_type_distribution": indicator_types,
        "operator_distribution": operators,
        "operand_distribution": operands,
        "unique_indicator_types": len(indicator_types),
    }
    
    return analysis

def print_analysis(analysis: Dict[str, Any]):
    """分析結果を表示"""
    print("\n" + "="*50)
    print("戦略多様性分析結果")
    print("="*50)
    
    if "error" in analysis:
        print(f"エラー: {analysis['error']}")
        return
    
    print(f"総戦略数: {analysis['total_strategies']}")
    print(f"総指標数: {analysis['total_indicators']}")
    print(f"戦略あたり平均指標数: {analysis['avg_indicators_per_strategy']:.2f}")
    print(f"ユニーク指標タイプ数: {analysis['unique_indicator_types']}")
    
    print("\n指標タイプ分布:")
    for indicator_type, count in sorted(analysis['indicator_type_distribution'].items()):
        percentage = (count / analysis['total_indicators']) * 100
        print(f"  {indicator_type}: {count}回 ({percentage:.1f}%)")
    
    print("\n演算子分布:")
    for operator, count in sorted(analysis['operator_distribution'].items()):
        print(f"  {operator}: {count}回")

def main():
    """メイン実行関数"""
    print("オートストラテジー生成テスト開始")
    print("="*50)
    
    # ランダム生成テスト
    print("\n1. ランダム戦略生成テスト")
    random_genes = test_random_gene_generation(5)
    
    # フォールバック生成テスト
    print("\n2. フォールバック戦略生成テスト")
    fallback_genes = test_fallback_generation(5)
    
    # 全体の多様性分析
    all_genes = random_genes + fallback_genes
    analysis = analyze_diversity(all_genes)
    print_analysis(analysis)
    
    print("\nテスト完了")

if __name__ == "__main__":
    main()
