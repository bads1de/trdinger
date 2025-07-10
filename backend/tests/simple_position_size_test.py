#!/usr/bin/env python3
"""
シンプルなポジションサイズテスト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.services.auto_strategy.models.gene_position_sizing import (
    PositionSizingGene,
    PositionSizingMethod,
    create_random_position_sizing_gene,
)


def simple_position_size_test():
    """シンプルなポジションサイズテスト"""
    print("=== シンプルなポジションサイズテスト ===\n")
    
    # 1. デフォルト設定
    print("1. デフォルト設定")
    default_gene = PositionSizingGene()
    print(f"  max_position_size: {default_gene.max_position_size}")
    print(f"  method: {default_gene.method}")
    print(f"  fixed_ratio: {default_gene.fixed_ratio}")
    
    # 2. ランダム生成
    print("\n2. ランダム生成（5回）")
    for i in range(5):
        try:
            random_gene = create_random_position_sizing_gene()
            print(f"  遺伝子 {i+1}:")
            print(f"    max_position_size: {random_gene.max_position_size}")
            print(f"    method: {random_gene.method}")
            print(f"    fixed_ratio: {random_gene.fixed_ratio}")
        except Exception as e:
            print(f"  遺伝子 {i+1}: エラー - {e}")
    
    # 3. 固定量方式での確認
    print("\n3. 固定量方式")
    fixed_quantity_gene = PositionSizingGene(
        method=PositionSizingMethod.FIXED_QUANTITY,
        fixed_quantity=10.0,  # 10.0枚
        max_position_size=50.0,
        enabled=True,
    )
    
    position_size = fixed_quantity_gene.calculate_position_size(
        account_balance=10000,
        current_price=50000
    )
    
    print(f"  fixed_quantity: {fixed_quantity_gene.fixed_quantity}")
    print(f"  max_position_size: {fixed_quantity_gene.max_position_size}")
    print(f"  計算結果: {position_size}")
    print(f"  制限されているか: {'はい' if position_size == fixed_quantity_gene.max_position_size else 'いいえ'}")


if __name__ == "__main__":
    simple_position_size_test()
