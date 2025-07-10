#!/usr/bin/env python3
"""
固定比率方式の計算を詳しく調査
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.services.auto_strategy.models.gene_position_sizing import (
    PositionSizingGene,
    PositionSizingMethod,
)
from app.core.services.auto_strategy.calculators.position_sizing_calculator import (
    PositionSizingCalculatorService,
)


def debug_fixed_ratio():
    """固定比率方式の計算を詳しく調査"""
    print("=== 固定比率方式計算調査 ===\n")
    
    # テストパラメータ
    account_balance = 10000.0
    current_price = 50000.0
    fixed_ratio = 2.0  # 200%
    
    # 1. PositionSizingGeneの直接計算
    print("1. PositionSizingGeneの直接計算")
    gene = PositionSizingGene(
        method=PositionSizingMethod.FIXED_RATIO,
        fixed_ratio=fixed_ratio,
        enabled=True,
    )
    
    direct_result = gene.calculate_position_size(
        account_balance=account_balance,
        current_price=current_price
    )
    
    print(f"  fixed_ratio: {gene.fixed_ratio}")
    print(f"  min_position_size: {gene.min_position_size}")
    print(f"  max_position_size: {gene.max_position_size}")
    print(f"  直接計算結果: {direct_result}")
    
    # 2. PositionSizingCalculatorServiceの計算
    print("\n2. PositionSizingCalculatorServiceの計算")
    calculator = PositionSizingCalculatorService()
    
    service_result = calculator.calculate_position_size(
        gene=gene,
        account_balance=account_balance,
        current_price=current_price,
        symbol="BTCUSDT",
    )
    
    print(f"  サービス計算結果: {service_result.position_size}")
    print(f"  計算詳細: {service_result.calculation_details}")
    print(f"  警告: {service_result.warnings}")
    
    # 3. 手動計算での確認
    print("\n3. 手動計算での確認")
    manual_calc = account_balance * fixed_ratio
    print(f"  手動計算 (account_balance * fixed_ratio): {manual_calc}")
    
    # 4. _apply_size_limitsの動作確認
    print("\n4. _apply_size_limitsの動作確認")
    test_values = [0.005, 0.01, 1.0, 10.0, 100.0, 20000.0]
    
    for value in test_values:
        limited_value = gene._apply_size_limits(value)
        print(f"  入力: {value} -> 制限後: {limited_value}")


if __name__ == "__main__":
    debug_fixed_ratio()
