#!/usr/bin/env python3
"""
TP/SL修正テスト

ショートポジションでのTP/SL計算が正しく動作するかをテストします。
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.core.services.auto_strategy.calculators.tpsl_calculator import TPSLCalculator
from app.core.services.auto_strategy.models.gene_tpsl import TPSLGene, TPSLMethod

def test_tpsl_calculation():
    """TP/SL計算のテスト"""
    print("=== TP/SL計算テスト ===")
    
    calculator = TPSLCalculator()
    current_price = 28000.0  # 現在価格
    stop_loss_pct = 0.03     # 3% SL
    take_profit_pct = 0.06   # 6% TP
    risk_management = {}
    
    print(f"現在価格: ${current_price:,.2f}")
    print(f"SL設定: {stop_loss_pct:.1%}")
    print(f"TP設定: {take_profit_pct:.1%}")
    
    # ロングポジションのテスト
    print(f"\n--- ロングポジション ---")
    sl_long, tp_long = calculator.calculate_legacy_tpsl_prices(
        current_price, stop_loss_pct, take_profit_pct, position_direction=1.0
    )
    print(f"SL価格: ${sl_long:,.2f} (現在価格より {((sl_long/current_price-1)*100):+.1f}%)")
    print(f"TP価格: ${tp_long:,.2f} (現在価格より {((tp_long/current_price-1)*100):+.1f}%)")
    print(f"backtesting.py要件: SL < 現在価格 < TP")
    print(f"実際: {sl_long:.2f} < {current_price:.2f} < {tp_long:.2f} = {sl_long < current_price < tp_long}")
    
    # ショートポジションのテスト
    print(f"\n--- ショートポジション ---")
    sl_short, tp_short = calculator.calculate_legacy_tpsl_prices(
        current_price, stop_loss_pct, take_profit_pct, position_direction=-1.0
    )
    print(f"SL価格: ${sl_short:,.2f} (現在価格より {((sl_short/current_price-1)*100):+.1f}%)")
    print(f"TP価格: ${tp_short:,.2f} (現在価格より {((tp_short/current_price-1)*100):+.1f}%)")
    print(f"backtesting.py要件: TP < 現在価格 < SL")
    print(f"実際: {tp_short:.2f} < {current_price:.2f} < {sl_short:.2f} = {tp_short < current_price < sl_short}")
    
    # 検証
    long_valid = sl_long < current_price < tp_long
    short_valid = tp_short < current_price < sl_short
    
    print(f"\n=== 検証結果 ===")
    print(f"ロングポジション: {'✅ 正常' if long_valid else '❌ 異常'}")
    print(f"ショートポジション: {'✅ 正常' if short_valid else '❌ 異常'}")
    
    if long_valid and short_valid:
        print("🎉 TP/SL計算修正が成功しました！")
    else:
        print("❌ TP/SL計算に問題があります")

def test_tpsl_gene_calculation():
    """TP/SL遺伝子計算のテスト"""
    print(f"\n=== TP/SL遺伝子計算テスト ===")

    calculator = TPSLCalculator()
    current_price = 28000.0

    # RISK_REWARD_RATIO方式のテスト（エラーで見られた値に近い設定）
    gene = TPSLGene(
        method=TPSLMethod.RISK_REWARD_RATIO,
        base_stop_loss=0.067,  # 6.7% SL
        risk_reward_ratio=1.8,  # 1.8倍のTP
    )

    print(f"遺伝子設定:")
    print(f"  方式: {gene.method}")
    print(f"  ベースSL: {gene.base_stop_loss:.1%}")
    print(f"  リスクリワード比: {gene.risk_reward_ratio}")

    # TP/SL値を計算
    tpsl_values = gene.calculate_tpsl_values()
    sl_pct = tpsl_values.get("stop_loss", 0.03)
    tp_pct = tpsl_values.get("take_profit", 0.06)

    print(f"計算されたパーセンテージ:")
    print(f"  SL: {sl_pct:.1%}")
    print(f"  TP: {tp_pct:.1%}")

    # ロングポジション
    sl_long, tp_long = calculator.calculate_tpsl_from_gene(current_price, gene, 1.0)
    print(f"\nロングポジション:")
    print(f"  SL: ${sl_long:.2f} ({((sl_long/current_price-1)*100):+.1f}%)")
    print(f"  TP: ${tp_long:.2f} ({((tp_long/current_price-1)*100):+.1f}%)")
    print(f"  要件: {sl_long:.2f} < {current_price:.2f} < {tp_long:.2f} = {sl_long < current_price < tp_long}")

    # ショートポジション
    sl_short, tp_short = calculator.calculate_tpsl_from_gene(current_price, gene, -1.0)
    print(f"\nショートポジション:")
    print(f"  SL: ${sl_short:.2f} ({((sl_short/current_price-1)*100):+.1f}%)")
    print(f"  TP: ${tp_short:.2f} ({((tp_short/current_price-1)*100):+.1f}%)")
    print(f"  要件: {tp_short:.2f} < {current_price:.2f} < {sl_short:.2f} = {tp_short < current_price < sl_short}")

def test_edge_cases():
    """エッジケースのテスト"""
    print(f"\n=== エッジケーステスト ===")

    calculator = TPSLCalculator()

    # 異なる価格でのテスト
    test_cases = [
        (50000.0, 0.02, 0.04),  # 高価格
        (1000.0, 0.05, 0.10),   # 低価格
        (28000.0, 0.01, 0.02),  # 小さなパーセンテージ
    ]

    for price, sl_pct, tp_pct in test_cases:
        print(f"\n価格: ${price:,.2f}, SL: {sl_pct:.1%}, TP: {tp_pct:.1%}")

        # ショートポジション
        sl_short, tp_short = calculator.calculate_legacy_tpsl_prices(
            price, sl_pct, tp_pct, position_direction=-1.0
        )

        short_valid = tp_short < price < sl_short
        print(f"  ショート: TP={tp_short:.2f} < 価格={price:.2f} < SL={sl_short:.2f} = {'✅' if short_valid else '❌'}")

def main():
    """メイン実行"""
    print("🔧 TP/SL修正テスト開始\n")
    
    try:
        test_tpsl_calculation()
        test_tpsl_gene_calculation()
        test_edge_cases()

        print("\n✅ テスト完了")
        print("\n📋 確認ポイント:")
        print("1. ロングポジション: SL < 現在価格 < TP")
        print("2. ショートポジション: TP < 現在価格 < SL")
        print("3. TP/SL遺伝子計算が正しく動作するか")
        print("4. backtesting.pyの要件を満たしているか")
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
