#!/usr/bin/env python3
"""
ポジションサイズの計算と表示をデバッグするテストスクリプト
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
from app.core.services.auto_strategy.calculators.position_sizing_helper import (
    PositionSizingHelper,
)
from app.core.services.auto_strategy.models.gene_strategy import StrategyGene


def test_position_sizing_precision():
    """ポジションサイズの精度をテスト"""
    print("=== ポジションサイズ精度テスト ===\n")
    
    # テストパラメータ
    account_balance = 10000.0
    current_price = 50000.0
    
    # 1. ボラティリティベース方式のテスト
    print("1. ボラティリティベース方式")
    volatility_gene = PositionSizingGene(
        method=PositionSizingMethod.VOLATILITY_BASED,
        risk_per_trade=0.02,  # 2%
        atr_multiplier=1.5,
        min_position_size=0.001,  # より小さい最小値
        max_position_size=50.0,
        enabled=True,
    )
    
    market_data = {
        "atr": 1000.0,  # ATR値
        "atr_pct": 1000.0 / current_price,  # 2%
    }
    
    calculator = PositionSizingCalculatorService()
    result = calculator.calculate_position_size(
        gene=volatility_gene,
        account_balance=account_balance,
        current_price=current_price,
        symbol="BTCUSDT",
        market_data=market_data,
    )
    
    print(f"  計算結果: {result.position_size}")
    print(f"  精度: {result.position_size:.8f}")
    print(f"  手法: {result.method_used}")
    print(f"  詳細: {result.calculation_details}")
    
    # 手動計算で検証
    risk_amount = account_balance * volatility_gene.risk_per_trade
    atr_pct = market_data["atr_pct"]
    volatility_factor = atr_pct * volatility_gene.atr_multiplier
    expected_size = risk_amount / (current_price * volatility_factor)
    
    print(f"  手動計算:")
    print(f"    リスク量: {risk_amount}")
    print(f"    ATR%: {atr_pct:.6f}")
    print(f"    ボラティリティ係数: {volatility_factor:.6f}")
    print(f"    期待サイズ: {expected_size:.8f}")
    
    # 2. 固定比率方式のテスト
    print("\n2. 固定比率方式")
    fixed_ratio_gene = PositionSizingGene(
        method=PositionSizingMethod.FIXED_RATIO,
        fixed_ratio=0.123456,  # 細かい小数点
        min_position_size=0.001,
        max_position_size=50.0,
        enabled=True,
    )
    
    result2 = calculator.calculate_position_size(
        gene=fixed_ratio_gene,
        account_balance=account_balance,
        current_price=current_price,
        symbol="BTCUSDT",
    )
    
    print(f"  計算結果: {result2.position_size}")
    print(f"  精度: {result2.position_size:.8f}")
    print(f"  手法: {result2.method_used}")
    
    # 3. PositionSizingHelperのテスト
    print("\n3. PositionSizingHelper経由のテスト")

    strategy_gene = StrategyGene(
        id="test_precision",
        indicators=[],
        entry_conditions=[],
        exit_conditions=[],
        risk_management={"position_size": 0.1},
        position_sizing_gene=volatility_gene,
    )

    # ログレベルを設定してエラーを確認
    import logging
    logging.basicConfig(level=logging.DEBUG)

    helper = PositionSizingHelper()

    # エラーをキャッチして詳細を表示
    try:
        helper_result = helper.calculate_position_size(
            strategy_gene, account_balance, current_price, None
        )
        print(f"  Helper結果: {helper_result}")
        print(f"  Helper精度: {helper_result:.8f}")
    except Exception as e:
        print(f"  Helper計算エラー: {e}")
        print(f"  エラータイプ: {type(e)}")
        import traceback
        traceback.print_exc()
    
    # 4. 丸め処理のテスト
    print("\n4. 丸め処理テスト")
    test_values = [1.123456789, 0.123456789, 10.987654321]
    
    for value in test_values:
        print(f"  元の値: {value:.9f}")
        print(f"  round(value, 3): {round(value, 3)}")
        print(f"  round(value, 6): {round(value, 6)}")
        print(f"  float(value): {float(value)}")
        print()


if __name__ == "__main__":
    test_position_sizing_precision()
