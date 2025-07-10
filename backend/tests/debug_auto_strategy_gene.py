#!/usr/bin/env python3
"""
AUTO_STRATEGYで生成される戦略遺伝子の設定を調査するスクリプト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.models.gene_position_sizing import create_random_position_sizing_gene


def debug_auto_strategy_gene_generation():
    """AUTO_STRATEGYで生成される戦略遺伝子を調査"""
    print("=== AUTO_STRATEGY戦略遺伝子生成調査 ===\n")
    
    # 1. デフォルトGAConfigでの生成
    print("1. デフォルトGAConfig")
    default_config = GAConfig.create_fast()
    print(f"  position_sizing_max_size_range: {default_config.position_sizing_max_size_range}")
    
    generator = RandomGeneGenerator(default_config)
    
    # 複数の戦略遺伝子を生成して確認
    for i in range(5):
        strategy_gene = generator.generate_strategy_gene()
        position_sizing_gene = strategy_gene.position_sizing_gene
        
        if position_sizing_gene:
            print(f"  戦略 {i+1}:")
            print(f"    method: {position_sizing_gene.method}")
            print(f"    max_position_size: {position_sizing_gene.max_position_size}")
            print(f"    fixed_ratio: {position_sizing_gene.fixed_ratio}")
            print(f"    risk_per_trade: {position_sizing_gene.risk_per_trade}")
        else:
            print(f"  戦略 {i+1}: position_sizing_gene が None")
    
    # 2. create_random_position_sizing_gene関数での直接生成
    print("\n2. create_random_position_sizing_gene関数での直接生成")
    for i in range(5):
        position_gene = create_random_position_sizing_gene()
        print(f"  遺伝子 {i+1}:")
        print(f"    method: {position_gene.method}")
        print(f"    max_position_size: {position_gene.max_position_size}")
        print(f"    fixed_ratio: {position_gene.fixed_ratio}")
        print(f"    risk_per_trade: {position_gene.risk_per_trade}")
    
    # 3. GAConfigの制約を確認
    print("\n3. GAConfigの制約確認")
    print(f"  position_sizing_method_constraints: {default_config.position_sizing_method_constraints}")
    print(f"  position_sizing_max_size_range: {default_config.position_sizing_max_size_range}")
    print(f"  position_sizing_fixed_ratio_range: {default_config.position_sizing_fixed_ratio_range}")
    print(f"  position_sizing_risk_per_trade_range: {default_config.position_sizing_risk_per_trade_range}")
    
    # 4. カスタムGAConfigでの生成（大きなmax_position_size）
    print("\n4. カスタムGAConfig（大きなmax_position_size）")
    custom_config = GAConfig.create_fast()
    custom_config.position_sizing_max_size_range = [10.0, 100.0]  # 大きな範囲に変更
    
    custom_generator = RandomGeneGenerator(custom_config)
    
    for i in range(3):
        strategy_gene = custom_generator.generate_strategy_gene()
        position_sizing_gene = strategy_gene.position_sizing_gene
        
        if position_sizing_gene:
            print(f"  戦略 {i+1}:")
            print(f"    method: {position_sizing_gene.method}")
            print(f"    max_position_size: {position_sizing_gene.max_position_size}")
            print(f"    fixed_ratio: {position_sizing_gene.fixed_ratio}")
            print(f"    risk_per_trade: {position_sizing_gene.risk_per_trade}")


if __name__ == "__main__":
    debug_auto_strategy_gene_generation()
