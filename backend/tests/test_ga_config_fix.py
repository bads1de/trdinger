#!/usr/bin/env python3
"""
GAConfig修正後のテスト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.models.gene_position_sizing import create_random_position_sizing_gene


def test_ga_config_fix():
    """GAConfig修正後のテスト"""
    print("=== GAConfig修正後のテスト ===\n")
    
    # 1. 修正後のGAConfig確認
    print("1. 修正後のGAConfig")
    config = GAConfig.create_fast()
    print(f"  position_sizing_max_size_range: {config.position_sizing_max_size_range}")
    
    # 2. GAConfigを使用したランダム生成
    print("\n2. GAConfigを使用したランダム生成（5回）")
    for i in range(5):
        try:
            random_gene = create_random_position_sizing_gene(config)
            print(f"  遺伝子 {i+1}:")
            print(f"    max_position_size: {random_gene.max_position_size:.6f}")
            print(f"    method: {random_gene.method}")
            print(f"    fixed_ratio: {random_gene.fixed_ratio:.6f}")
            print(f"    risk_per_trade: {random_gene.risk_per_trade:.6f}")
        except Exception as e:
            print(f"  遺伝子 {i+1}: エラー - {e}")
    
    # 3. GAConfigなしでの生成（比較用）
    print("\n3. GAConfigなしでの生成（比較用）")
    for i in range(3):
        try:
            random_gene = create_random_position_sizing_gene()
            print(f"  遺伝子 {i+1}:")
            print(f"    max_position_size: {random_gene.max_position_size:.6f}")
            print(f"    method: {random_gene.method}")
        except Exception as e:
            print(f"  遺伝子 {i+1}: エラー - {e}")


if __name__ == "__main__":
    test_ga_config_fix()
