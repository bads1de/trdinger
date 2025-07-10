#!/usr/bin/env python3
"""
max_position_size無制限化のテスト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.services.auto_strategy.models.gene_position_sizing import (
    PositionSizingGene,
    PositionSizingMethod,
    create_random_position_sizing_gene,
)
from app.core.services.auto_strategy.calculators.position_sizing_calculator import (
    PositionSizingCalculatorService,
)
from app.core.services.auto_strategy.models.ga_config import GAConfig


def test_unlimited_position_size():
    """max_position_size無制限化のテスト"""
    print("=== max_position_size無制限化テスト ===\n")
    
    # 1. デフォルト設定確認
    print("1. デフォルト設定")
    default_gene = PositionSizingGene()
    print(f"  max_position_size: {default_gene.max_position_size}")
    print(f"  max_position_sizeが無限大か: {default_gene.max_position_size == float('inf')}")
    
    # 2. 大きなポジションサイズでの計算テスト
    print("\n2. 大きなポジションサイズでの計算テスト")
    
    # 固定比率で大きな値を設定
    large_ratio_gene = PositionSizingGene(
        method=PositionSizingMethod.FIXED_RATIO,
        fixed_ratio=2.0,  # 200%（従来なら制限される値）
        enabled=True,
    )
    
    calculator = PositionSizingCalculatorService()
    result = calculator.calculate_position_size(
        gene=large_ratio_gene,
        account_balance=10000.0,
        current_price=50000.0,
        symbol="BTCUSDT",
    )
    
    print(f"  fixed_ratio: {large_ratio_gene.fixed_ratio}")
    print(f"  max_position_size: {large_ratio_gene.max_position_size}")
    print(f"  期待値 (account_balance * fixed_ratio): {10000.0 * large_ratio_gene.fixed_ratio}")
    print(f"  計算結果: {result.position_size}")
    print(f"  制限されているか: {'はい' if result.position_size != 10000.0 * large_ratio_gene.fixed_ratio else 'いいえ'}")
    
    # 3. 固定枚数で大きな値を設定
    print("\n3. 固定枚数で大きな値を設定")
    
    large_quantity_gene = PositionSizingGene(
        method=PositionSizingMethod.FIXED_QUANTITY,
        fixed_quantity=100.0,  # 100枚（従来なら制限される値）
        enabled=True,
    )
    
    result2 = calculator.calculate_position_size(
        gene=large_quantity_gene,
        account_balance=10000.0,
        current_price=50000.0,
        symbol="BTCUSDT",
    )
    
    print(f"  fixed_quantity: {large_quantity_gene.fixed_quantity}")
    print(f"  max_position_size: {large_quantity_gene.max_position_size}")
    print(f"  計算結果: {result2.position_size}")
    print(f"  制限されているか: {'はい' if result2.position_size != large_quantity_gene.fixed_quantity else 'いいえ'}")
    
    # 4. ランダム生成での確認
    print("\n4. ランダム生成での確認")
    config = GAConfig.create_fast()
    
    for i in range(3):
        random_gene = create_random_position_sizing_gene(config)
        print(f"  遺伝子 {i+1}:")
        print(f"    method: {random_gene.method}")
        print(f"    max_position_size: {random_gene.max_position_size}")
        print(f"    max_position_sizeが無限大か: {random_gene.max_position_size == float('inf')}")
    
    # 5. バリデーションテスト
    print("\n5. バリデーションテスト")
    
    # 正常なケース
    valid_gene = PositionSizingGene(
        min_position_size=0.01,
        max_position_size=float('inf'),
    )
    is_valid, errors = valid_gene.validate()
    print(f"  正常なケース: {'✅' if is_valid else '❌'}")
    if errors:
        print(f"    エラー: {errors}")
    
    # 異常なケース（min > max）
    invalid_gene = PositionSizingGene(
        min_position_size=10.0,
        max_position_size=5.0,  # min_position_sizeより小さい
    )
    is_valid, errors = invalid_gene.validate()
    print(f"  異常なケース: {'❌' if not is_valid else '✅'}")
    if errors:
        print(f"    エラー: {errors}")


if __name__ == "__main__":
    test_unlimited_position_size()
