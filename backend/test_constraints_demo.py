#!/usr/bin/env python3
"""
制約エンジンのデモンストレーション
"""

import sys
sys.path.append('.')

from app.core.services.indicators.constraints import constraint_engine, OrderConstraint, RangeConstraint

def main():
    print("=== 制約エンジンのデモンストレーション ===\n")

    # MACD制約のテスト
    print("1. MACD制約のテスト")
    params = {'fast_period': 26, 'slow_period': 12, 'signal_period': 9}
    print(f"調整前: {params}")
    result = constraint_engine.apply_constraints('MACD', params)
    print(f"調整後: {result}")
    print(f"制約検証: {constraint_engine.validate_constraints('MACD', result)}")
    print()

    # Stochastic制約のテスト
    print("2. Stochastic制約のテスト")
    stoch_params = {'slowk_matype': 15, 'slowd_matype': 20}
    print(f"調整前: {stoch_params}")
    stoch_result = constraint_engine.apply_constraints('STOCH', stoch_params)
    print(f"調整後: {stoch_result}")
    print(f"制約検証: {constraint_engine.validate_constraints('STOCH', stoch_result)}")
    print()

    # MACDEXT制約のテスト
    print("3. MACDEXT制約のテスト")
    macdext_params = {
        'fast_period': 30, 
        'slow_period': 15, 
        'signal_period': 9,
        'fast_ma_type': 15,
        'slow_ma_type': 20,
        'signal_ma_type': 25
    }
    print(f"調整前: {macdext_params}")
    macdext_result = constraint_engine.apply_constraints('MACDEXT', macdext_params)
    print(f"調整後: {macdext_result}")
    print(f"制約検証: {constraint_engine.validate_constraints('MACDEXT', macdext_result)}")
    print()

    # 登録されている制約の確認
    print("4. 登録されている制約")
    indicators = constraint_engine.list_indicators()
    print(f"制約が登録されているインディケーター: {indicators}")
    
    for indicator in indicators:
        constraints = constraint_engine.get_constraints(indicator)
        print(f"{indicator}: {len(constraints)}個の制約")
        for constraint in constraints:
            print(f"  - {constraint.get_description()}")
    print()

    print("=== デモ完了 ===")

if __name__ == "__main__":
    main()
