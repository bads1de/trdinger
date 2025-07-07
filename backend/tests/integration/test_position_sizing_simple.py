"""
ポジションサイジングの簡単な動作確認テスト
"""

import sys
import os

# テスト対象のモジュールをインポートするためのパス設定
backend_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, backend_path)

from app.core.services.auto_strategy.models.position_sizing_gene import (
    PositionSizingGene,
    PositionSizingMethod,
    create_random_position_sizing_gene,
)
from app.core.services.auto_strategy.calculators.position_sizing_calculator import (
    PositionSizingCalculatorService,
)


def test_position_sizing_basic():
    """基本的なポジションサイジングテスト"""
    print("=== ポジションサイジング基本テスト ===")

    # 1. 固定比率方式のテスト
    print("\n1. 固定比率方式のテスト")
    gene = PositionSizingGene(
        method=PositionSizingMethod.FIXED_RATIO,
        fixed_ratio=0.2,
        min_position_size=0.01,
        max_position_size=5.0,
    )

    calculator = PositionSizingCalculatorService()
    result = calculator.calculate_position_size(
        gene=gene,
        account_balance=10000.0,
        current_price=50000.0,
        symbol="BTCUSDT",
    )

    print(f"  口座残高: 10,000")
    print(f"  固定比率: 20%")
    print(f"  計算結果: {result.position_size}")
    print(f"  使用手法: {result.method_used}")
    print(f"  信頼度: {result.confidence_score:.2f}")

    # 最大サイズ制限により5.0に制限される
    expected_size = min(10000.0 * 0.2, 5.0)  # min(2000.0, 5.0) = 5.0
    assert result.position_size == expected_size
    assert result.method_used == "fixed_ratio"

    # 2. ボラティリティベース方式のテスト
    print("\n2. ボラティリティベース方式のテスト")
    gene = PositionSizingGene(
        method=PositionSizingMethod.VOLATILITY_BASED,
        atr_multiplier=2.0,
        risk_per_trade=0.02,
        min_position_size=0.01,
        max_position_size=5.0,
    )

    market_data = {"atr": 1000.0, "atr_source": "test"}

    result = calculator.calculate_position_size(
        gene=gene,
        account_balance=10000.0,
        current_price=50000.0,
        symbol="BTCUSDT",
        market_data=market_data,
    )

    print(f"  口座残高: 10,000")
    print(f"  リスク率: 2%")
    print(f"  ATR: 1,000")
    print(f"  ATR倍率: 2.0")
    print(f"  計算結果: {result.position_size}")
    print(f"  使用手法: {result.method_used}")
    print(f"  信頼度: {result.confidence_score:.2f}")

    expected = 200.0 / (1000.0 * 2.0)  # risk_amount / (atr * multiplier)
    assert result.position_size == expected
    assert result.method_used == "volatility_based"

    # 3. 固定枚数方式のテスト
    print("\n3. 固定枚数方式のテスト")
    gene = PositionSizingGene(
        method=PositionSizingMethod.FIXED_QUANTITY,
        fixed_quantity=3.0,
        min_position_size=0.01,
        max_position_size=5.0,
    )

    result = calculator.calculate_position_size(
        gene=gene,
        account_balance=10000.0,
        current_price=50000.0,
        symbol="BTCUSDT",
    )

    print(f"  固定枚数: 3.0")
    print(f"  計算結果: {result.position_size}")
    print(f"  使用手法: {result.method_used}")
    print(f"  信頼度: {result.confidence_score:.2f}")

    assert result.position_size == 3.0
    assert result.method_used == "fixed_quantity"

    print("\n✅ 全ての基本テストが成功しました！")


def test_position_sizing_gene_operations():
    """ポジションサイジング遺伝子の操作テスト"""
    print("\n=== ポジションサイジング遺伝子操作テスト ===")

    # 1. ランダム遺伝子生成
    print("\n1. ランダム遺伝子生成テスト")
    gene = create_random_position_sizing_gene()

    print(f"  生成された手法: {gene.method.value}")
    print(f"  固定比率: {gene.fixed_ratio:.3f}")
    print(f"  固定枚数: {gene.fixed_quantity:.3f}")
    print(f"  優先度: {gene.priority:.3f}")

    # バリデーション
    is_valid, errors = gene.validate()
    print(f"  バリデーション: {'✅ 有効' if is_valid else '❌ 無効'}")
    if errors:
        for error in errors:
            print(f"    エラー: {error}")

    assert is_valid is True
    assert len(errors) == 0

    # 2. 辞書変換テスト
    print("\n2. 辞書変換テスト")
    gene_dict = gene.to_dict()
    restored_gene = PositionSizingGene.from_dict(gene_dict)

    print(f"  元の手法: {gene.method.value}")
    print(f"  復元後手法: {restored_gene.method.value}")
    print(f"  元の固定比率: {gene.fixed_ratio:.3f}")
    print(f"  復元後固定比率: {restored_gene.fixed_ratio:.3f}")

    assert gene.method == restored_gene.method
    assert gene.fixed_ratio == restored_gene.fixed_ratio
    assert gene.fixed_quantity == restored_gene.fixed_quantity

    print("\n✅ 全ての遺伝子操作テストが成功しました！")


def test_position_sizing_error_handling():
    """ポジションサイジングのエラーハンドリングテスト"""
    print("\n=== エラーハンドリングテスト ===")

    calculator = PositionSizingCalculatorService()

    # 1. 無効な遺伝子のテスト
    print("\n1. 無効な遺伝子のテスト")
    result = calculator.calculate_position_size(
        gene=None,
        account_balance=10000.0,
        current_price=50000.0,
        symbol="BTCUSDT",
    )

    print(f"  計算結果: {result.position_size}")
    print(f"  警告数: {len(result.warnings)}")
    print(f"  信頼度: {result.confidence_score:.2f}")

    assert result.position_size == 0.01  # エラー時の最小サイズ
    assert len(result.warnings) > 0
    assert result.confidence_score == 0.0

    # 2. 負の口座残高のテスト
    print("\n2. 負の口座残高のテスト")
    gene = PositionSizingGene(method=PositionSizingMethod.FIXED_RATIO)

    result = calculator.calculate_position_size(
        gene=gene,
        account_balance=-1000.0,
        current_price=50000.0,
        symbol="BTCUSDT",
    )

    print(f"  計算結果: {result.position_size}")
    print(f"  警告数: {len(result.warnings)}")

    assert result.position_size == 0.01  # エラー時の最小サイズ
    assert len(result.warnings) > 0

    # 3. 無効化された遺伝子のテスト
    print("\n3. 無効化された遺伝子のテスト")
    gene = PositionSizingGene(
        method=PositionSizingMethod.FIXED_RATIO,
        enabled=False,
        min_position_size=0.05,
    )

    result = calculator.calculate_position_size(
        gene=gene,
        account_balance=10000.0,
        current_price=50000.0,
        symbol="BTCUSDT",
    )

    print(f"  計算結果: {result.position_size}")
    print(f"  使用手法: {result.method_used}")

    # 無効化された遺伝子の場合、従来のrisk_managementが使用される
    # デフォルトのposition_size（0.1）が使用され、max_position_size（1.0）で制限される
    assert result.position_size == 1.0  # min(10000 * 0.1, 1.0) = 1.0

    print("\n✅ 全てのエラーハンドリングテストが成功しました！")


def main():
    """メイン関数"""
    print("ポジションサイジングシステム動作確認テスト開始")
    print("=" * 50)

    try:
        test_position_sizing_basic()
        test_position_sizing_gene_operations()
        test_position_sizing_error_handling()

        print("\n" + "=" * 50)
        print("🎉 全てのテストが成功しました！")
        print("ポジションサイジングシステムは正常に動作しています。")

    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
