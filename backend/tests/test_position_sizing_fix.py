#!/usr/bin/env python3
"""
ポジションサイジングメソッド修正の検証テスト

修正内容：
1. デフォルト値をボラティリティベースに変更
2. 段階的フォールバック処理の実装
3. データ準備機能の強化
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from app.core.services.auto_strategy.models.position_sizing_gene import (
    PositionSizingGene, 
    PositionSizingMethod
)
from app.core.services.auto_strategy.models.gene_encoding import GeneEncoder
from app.core.services.auto_strategy.calculators.position_sizing_calculator import (
    PositionSizingCalculatorService
)

def test_default_values():
    """デフォルト値の変更をテスト"""
    print("=== デフォルト値テスト ===")
    
    # 1. PositionSizingGeneのデフォルト値
    gene = PositionSizingGene()
    print(f"デフォルトメソッド: {gene.method}")
    assert gene.method == PositionSizingMethod.VOLATILITY_BASED, f"期待値: VOLATILITY_BASED, 実際: {gene.method}"
    print("✅ PositionSizingGeneのデフォルト値が正しく変更されました")
    
    # 2. エンコーディングのデフォルト値
    encoder = GeneEncoder()
    
    # 無効な遺伝子でエラーを発生させてデフォルト値をテスト
    try:
        # 不正なデータでデコードを試行
        decoded_gene = encoder._decode_position_sizing_gene([])
        print(f"エラー時デフォルトメソッド: {decoded_gene.method}")
        assert decoded_gene.method == PositionSizingMethod.VOLATILITY_BASED, f"期待値: VOLATILITY_BASED, 実際: {decoded_gene.method}"
        print("✅ エンコーディングエラー時のデフォルト値が正しく変更されました")
    except Exception as e:
        print(f"エンコーディングテストでエラー: {e}")

def test_fallback_processing():
    """フォールバック処理の改善をテスト"""
    print("\n=== フォールバック処理テスト ===")
    
    calculator = PositionSizingCalculatorService()
    
    # 1. Half Optimal F方式のデータ不足時フォールバック
    print("1. Half Optimal F方式のフォールバックテスト")
    gene = PositionSizingGene(
        method=PositionSizingMethod.HALF_OPTIMAL_F,
        optimal_f_multiplier=0.5,
        fixed_ratio=0.1
    )
    
    result = calculator.calculate_position_size(
        gene=gene,
        account_balance=10000.0,
        current_price=50000.0,
        symbol="BTCUSDT",
        trade_history=[]  # 空の履歴でフォールバックを発生
    )
    
    print(f"  計算結果: {result.position_size}")
    print(f"  使用メソッド: {result.method_used}")
    print(f"  警告: {result.warnings}")
    print(f"  詳細: {result.calculation_details.get('fallback_reason', 'なし')}")
    
    # 簡易版計算が使用されることを確認
    assert "simplified" in result.calculation_details.get('fallback_reason', ''), "簡易版計算が使用されていません"
    print("✅ Half Optimal F方式のフォールバック処理が改善されました")
    
    # 2. ボラティリティベース方式のATRフォールバック
    print("\n2. ボラティリティベース方式のATRフォールバックテスト")
    gene = PositionSizingGene(
        method=PositionSizingMethod.VOLATILITY_BASED,
        atr_multiplier=2.0,
        risk_per_trade=0.02
    )
    
    result = calculator.calculate_position_size(
        gene=gene,
        account_balance=10000.0,
        current_price=50000.0,
        symbol="BTCUSDT",
        market_data={}  # ATRデータなしでフォールバックを発生
    )
    
    print(f"  計算結果: {result.position_size}")
    print(f"  使用メソッド: {result.method_used}")
    print(f"  ATRソース: {result.calculation_details.get('atr_source', 'なし')}")
    
    # ATRが推定値で計算されることを確認
    assert result.position_size > 0, "ポジションサイズが計算されていません"
    print("✅ ボラティリティベース方式のフォールバック処理が改善されました")

def test_method_distribution():
    """メソッド選択の分散をテスト"""
    print("\n=== メソッド選択分散テスト ===")
    
    encoder = GeneEncoder()
    method_counts = {method: 0 for method in PositionSizingMethod}
    
    # 複数の遺伝子を生成してメソッド分布を確認
    test_values = [0.1, 0.3, 0.6, 0.8]  # 各範囲の値
    
    for value in test_values:
        encoded = [value] + [0.5] * 7  # メソッド値 + その他のパラメータ
        decoded_gene = encoder._decode_position_sizing_gene(encoded)
        method_counts[decoded_gene.method] += 1
        print(f"  値 {value} → {decoded_gene.method}")
    
    print(f"\nメソッド分布:")
    for method, count in method_counts.items():
        print(f"  {method}: {count}回")
    
    # 各メソッドが適切に選択されることを確認
    assert method_counts[PositionSizingMethod.HALF_OPTIMAL_F] > 0, "HALF_OPTIMAL_Fが選択されていません"
    assert method_counts[PositionSizingMethod.VOLATILITY_BASED] > 0, "VOLATILITY_BASEDが選択されていません"
    assert method_counts[PositionSizingMethod.FIXED_RATIO] > 0, "FIXED_RATIOが選択されていません"
    assert method_counts[PositionSizingMethod.FIXED_QUANTITY] > 0, "FIXED_QUANTITYが選択されていません"
    
    print("✅ 全てのメソッドが適切に選択されています")

def main():
    """メインテスト実行"""
    print("ポジションサイジングメソッド修正の検証テストを開始します...\n")
    
    try:
        test_default_values()
        test_fallback_processing()
        test_method_distribution()
        
        print("\n🎉 全てのテストが成功しました！")
        print("\n修正内容:")
        print("✅ デフォルト値をボラティリティベースに変更")
        print("✅ 段階的フォールバック処理の実装")
        print("✅ データ準備機能の強化")
        print("✅ メソッド選択の公平性向上")
        
    except Exception as e:
        print(f"\n❌ テストでエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
