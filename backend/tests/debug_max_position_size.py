#!/usr/bin/env python3
"""
max_position_sizeの制限を調査するスクリプト
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


def debug_max_position_size():
    """max_position_sizeの制限を調査"""
    print("=== max_position_size制限調査 ===\n")
    
    # テストパラメータ
    account_balance = 10000.0
    current_price = 50000.0
    
    # 1. デフォルト設定での計算
    print("1. デフォルト設定")
    default_gene = PositionSizingGene()
    print(f"  デフォルトmax_position_size: {default_gene.max_position_size}")
    print(f"  デフォルトmethod: {default_gene.method}")
    print(f"  デフォルトfixed_ratio: {default_gene.fixed_ratio}")
    
    calculator = PositionSizingCalculatorService()
    result = calculator.calculate_position_size(
        gene=default_gene,
        account_balance=account_balance,
        current_price=current_price,
        symbol="BTCUSDT",
    )
    
    print(f"  計算結果: {result.position_size}")
    print(f"  制限前の計算値: {result.calculation_details}")
    
    # 2. 大きなmax_position_sizeでの計算
    print("\n2. 大きなmax_position_size設定")
    large_max_gene = PositionSizingGene(
        method=PositionSizingMethod.FIXED_RATIO,
        fixed_ratio=0.5,  # 50%
        max_position_size=100.0,  # 大きな上限
        enabled=True,
    )
    
    result2 = calculator.calculate_position_size(
        gene=large_max_gene,
        account_balance=account_balance,
        current_price=current_price,
        symbol="BTCUSDT",
    )
    
    print(f"  max_position_size: {large_max_gene.max_position_size}")
    print(f"  fixed_ratio: {large_max_gene.fixed_ratio}")
    print(f"  期待値 (account_balance * fixed_ratio): {account_balance * large_max_gene.fixed_ratio}")
    print(f"  計算結果: {result2.position_size}")
    
    # 3. ボラティリティベースでの計算
    print("\n3. ボラティリティベース（大きなmax_position_size）")
    volatility_gene = PositionSizingGene(
        method=PositionSizingMethod.VOLATILITY_BASED,
        risk_per_trade=0.02,  # 2%
        atr_multiplier=1.5,
        max_position_size=100.0,  # 大きな上限
        enabled=True,
    )
    
    market_data = {
        "atr": 1000.0,  # ATR値
        "atr_pct": 1000.0 / current_price,  # 2%
    }
    
    result3 = calculator.calculate_position_size(
        gene=volatility_gene,
        account_balance=account_balance,
        current_price=current_price,
        symbol="BTCUSDT",
        market_data=market_data,
    )
    
    print(f"  max_position_size: {volatility_gene.max_position_size}")
    print(f"  計算結果: {result3.position_size}")
    print(f"  詳細: {result3.calculation_details}")
    
    # 4. 制限の確認
    print("\n4. 制限の動作確認")
    
    # 制限前の値を手動計算
    risk_amount = account_balance * volatility_gene.risk_per_trade
    atr_pct = market_data["atr_pct"]
    volatility_factor = atr_pct * volatility_gene.atr_multiplier
    raw_size = risk_amount / (current_price * volatility_factor)
    
    print(f"  制限前の計算値: {raw_size:.8f}")
    print(f"  min_position_size: {volatility_gene.min_position_size}")
    print(f"  max_position_size: {volatility_gene.max_position_size}")
    
    # 制限適用後
    limited_size = max(volatility_gene.min_position_size, min(raw_size, volatility_gene.max_position_size))
    print(f"  制限適用後: {limited_size:.8f}")
    print(f"  実際の結果: {result3.position_size:.8f}")
    
    # 5. 現在の設定での問題確認
    print("\n5. 現在の設定での問題")
    current_gene = PositionSizingGene(
        method=PositionSizingMethod.VOLATILITY_BASED,
        risk_per_trade=0.02,
        atr_multiplier=2.0,
        max_position_size=10.0,  # デフォルト値
        enabled=True,
    )
    
    result4 = calculator.calculate_position_size(
        gene=current_gene,
        account_balance=account_balance,
        current_price=current_price,
        symbol="BTCUSDT",
        market_data=market_data,
    )
    
    print(f"  デフォルトmax_position_size: {current_gene.max_position_size}")
    print(f"  計算結果: {result4.position_size}")
    print(f"  制限されているか: {'はい' if result4.position_size == current_gene.max_position_size else 'いいえ'}")


if __name__ == "__main__":
    debug_max_position_size()
