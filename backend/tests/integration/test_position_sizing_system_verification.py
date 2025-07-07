"""
Position Sizingシステムの動作確認テスト

従来のposition_size_rangeが削除され、新しいPosition Sizingシステムが
正常に動作していることを確認するテストです。
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
from app.core.services.auto_strategy.models.strategy_gene import StrategyGene
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory


def test_position_sizing_system_integration():
    """Position Sizingシステムの統合動作確認"""
    print("=== Position Sizingシステム統合テスト ===")

    # 1. GAConfigでposition_size_rangeが削除されていることを確認
    print("\n1. GAConfig確認")
    config = GAConfig()

    # position_size_rangeが存在しないことを確認
    assert not hasattr(
        config, "position_size_range"
    ), "position_size_rangeが削除されていません"

    # 新しいPosition Sizing関連パラメータが存在することを確認
    assert hasattr(
        config, "position_sizing_method_constraints"
    ), "position_sizing_method_constraintsが存在しません"
    assert hasattr(
        config, "position_sizing_fixed_ratio_range"
    ), "position_sizing_fixed_ratio_rangeが存在しません"

    print("  ✅ GAConfigからposition_size_rangeが正常に削除されています")
    print(
        f"  ✅ Position Sizing制約が設定されています: {len(config.position_sizing_method_constraints)}個の手法"
    )

    # 2. RandomGeneGeneratorでPosition Sizingが動作することを確認
    print("\n2. RandomGeneGenerator確認")
    generator = RandomGeneGenerator(config)

    # 戦略遺伝子を生成
    strategy_gene = generator.generate_random_gene()

    # position_sizing_geneが生成されていることを確認
    assert hasattr(
        strategy_gene, "position_sizing_gene"
    ), "position_sizing_geneが生成されていません"

    if strategy_gene.position_sizing_gene:
        print(
            f"  ✅ Position Sizing遺伝子が生成されました: {strategy_gene.position_sizing_gene.method.value}"
        )
        print(f"  ✅ 固定比率: {strategy_gene.position_sizing_gene.fixed_ratio:.3f}")
    else:
        print("  ⚠️ Position Sizing遺伝子がNullです（ランダム生成のため正常）")

    # 3. StrategyFactoryでPosition Sizingが動作することを確認
    print("\n3. StrategyFactory確認")

    # Position Sizing遺伝子を持つ戦略遺伝子を作成
    position_sizing_gene = PositionSizingGene(
        method=PositionSizingMethod.FIXED_RATIO,
        fixed_ratio=0.15,
        min_position_size=0.01,
        max_position_size=2.0,
        enabled=True,
    )

    test_strategy_gene = StrategyGene(
        id="test_position_sizing",
        indicators=[],
        entry_conditions=[],
        exit_conditions=[],
        risk_management={"position_size": 0.1},  # 従来値（使用されない）
        position_sizing_gene=position_sizing_gene,
    )

    factory = StrategyFactory()

    # ポジションサイズ計算をテスト
    calculated_size = factory._calculate_position_size(
        test_strategy_gene, account_balance=10000.0, current_price=50000.0, data=None
    )

    print(f"  ✅ StrategyFactoryでポジションサイズが計算されました: {calculated_size}")

    # Position Sizing遺伝子の設定に基づいて計算されていることを確認
    expected_size = min(10000.0 * 0.15, 2.0)  # min(1500.0, 2.0) = 2.0
    assert (
        calculated_size == expected_size
    ), f"期待値: {expected_size}, 実際: {calculated_size}"

    print(f"  ✅ 計算結果が期待値と一致しています: {expected_size}")

    # 4. Position Sizing無効時のフォールバック確認
    print("\n4. フォールバック動作確認")

    # Position Sizing遺伝子を無効化
    disabled_gene = PositionSizingGene(enabled=False)

    test_strategy_gene_disabled = StrategyGene(
        id="test_fallback",
        indicators=[],
        entry_conditions=[],
        exit_conditions=[],
        risk_management={"position_size": 0.2},  # フォールバック値
        position_sizing_gene=disabled_gene,
    )

    fallback_size = factory._calculate_position_size(
        test_strategy_gene_disabled,
        account_balance=10000.0,
        current_price=50000.0,
        data=None,
    )

    print(f"  ✅ フォールバック動作確認: {fallback_size}")

    # 従来のrisk_managementが使用されることを確認
    expected_fallback = max(
        0.01, min(0.2, 1.0)
    )  # min(risk_management.position_size, max_default)
    assert (
        fallback_size == expected_fallback
    ), f"フォールバック期待値: {expected_fallback}, 実際: {fallback_size}"

    print(f"  ✅ フォールバック値が正常です: {expected_fallback}")

    print("\n✅ Position Sizingシステムの統合テストが全て成功しました！")


def test_position_sizing_calculator_service():
    """Position Sizing計算サービスの動作確認"""
    print("\n=== Position Sizing計算サービステスト ===")

    calculator = PositionSizingCalculatorService()

    # 各手法のテスト
    methods_to_test = [
        (PositionSizingMethod.FIXED_RATIO, {"fixed_ratio": 0.2}),
        (PositionSizingMethod.FIXED_QUANTITY, {"fixed_quantity": 1.5}),
        (
            PositionSizingMethod.VOLATILITY_BASED,
            {"risk_per_trade": 0.03, "atr_multiplier": 2.5},
        ),
    ]

    for method, params in methods_to_test:
        print(f"\n{method.value}方式のテスト:")

        gene = PositionSizingGene(method=method, **params)

        market_data = (
            {"atr": 800.0, "atr_source": "test"}
            if method == PositionSizingMethod.VOLATILITY_BASED
            else None
        )

        result = calculator.calculate_position_size(
            gene=gene,
            account_balance=10000.0,
            current_price=40000.0,
            symbol="BTCUSDT",
            market_data=market_data,
        )

        print(f"  計算結果: {result.position_size}")
        print(f"  使用手法: {result.method_used}")
        print(f"  信頼度: {result.confidence_score:.2f}")
        print(f"  警告数: {len(result.warnings)}")

        # 基本的な妥当性チェック
        assert result.position_size > 0, f"{method.value}で無効なポジションサイズ"
        assert result.method_used == method.value, f"使用手法が一致しません"
        assert 0.0 <= result.confidence_score <= 1.0, f"信頼度が範囲外"

        print(f"  ✅ {method.value}方式が正常に動作しています")

    print("\n✅ 全ての計算手法が正常に動作しています！")


def test_legacy_system_removal():
    """従来システムの削除確認"""
    print("\n=== 従来システム削除確認テスト ===")

    # 1. GAConfigでposition_size_rangeが削除されていることを再確認
    config = GAConfig()
    config_dict = config.__dict__

    position_size_related = [
        key
        for key in config_dict.keys()
        if "position_size" in key and "position_sizing" not in key
    ]

    print(f"従来のposition_size関連パラメータ: {position_size_related}")
    assert (
        len(position_size_related) == 0
    ), f"従来のposition_size関連パラメータが残っています: {position_size_related}"

    # 2. 新しいposition_sizing関連パラメータが存在することを確認
    position_sizing_related = [
        key for key in config_dict.keys() if "position_sizing" in key
    ]

    print(f"新しいposition_sizing関連パラメータ: {len(position_sizing_related)}個")
    assert (
        len(position_sizing_related) >= 8
    ), f"position_sizing関連パラメータが不足: {len(position_sizing_related)}個"

    print("  ✅ 従来システムが正常に削除され、新システムが導入されています")

    # 3. RandomGeneGeneratorで固定値が使用されることを確認
    generator = RandomGeneGenerator(config)
    risk_management = generator._generate_legacy_risk_management()

    print(f"従来リスク管理設定: {risk_management}")
    assert risk_management["position_size"] == 0.1, "固定値が設定されていません"

    print("  ✅ RandomGeneGeneratorで固定値が使用されています")

    print("\n✅ 従来システムの削除が正常に完了しています！")


def main():
    """メイン関数"""
    print("Position Sizingシステム動作確認テスト開始")
    print("=" * 60)

    try:
        test_position_sizing_system_integration()
        test_position_sizing_calculator_service()
        test_legacy_system_removal()

        print("\n" + "=" * 60)
        print("🎉 全てのテストが成功しました！")
        print(
            "Position Sizingシステムが正常に動作し、従来システムが適切に削除されています。"
        )

    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
