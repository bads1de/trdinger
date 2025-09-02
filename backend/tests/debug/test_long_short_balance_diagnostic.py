"""
ロング・ショート戦略バランス診断テスト

オートストラテジーでロングオンリー問題を診断するためのテスト
"""
import pytest
import logging
from typing import List

from app.services.auto_strategy.generators.condition_generator import ConditionGenerator
from app.services.auto_strategy.models.strategy_models import IndicatorGene
from app.services.auto_strategy.config.constants import IndicatorType


class TestLongShortBalanceDiagnostic:
    """ロング・ショートバランス診断テスト"""

    @pytest.fixture
    def generator(self):
        """ConditionGeneratorインスタンス"""
        return ConditionGenerator(enable_smart_generation=True)

    @pytest.fixture
    def statistics_indicators(self) -> List[IndicatorGene]:
        """統計指標セット"""
        return [
            IndicatorGene(type="RSI", enabled=True),
            IndicatorGene(type="STOCH", enabled=True),
            IndicatorGene(type="CCI", enabled=True),
        ]

    @pytest.fixture
    def pattern_indicators(self) -> List[IndicatorGene]:
        """パターンマッチ指標セット"""
        return [
            IndicatorGene(type="SMA", enabled=True),
        ]

    @pytest.fixture
    def trend_indicators(self) -> List[IndicatorGene]:
        """トレンド指標セット"""
        return [
            IndicatorGene(type="SMA", enabled=True),
            IndicatorGene(type="EMA", enabled=True),
            IndicatorGene(type="MACD", enabled=True),
        ]

    @pytest.fixture
    def momentum_indicators(self) -> List[IndicatorGene]:
        """モメンタム指標セット"""
        return [
            IndicatorGene(type="ROC", enabled=True),
            IndicatorGene(type="MFI", enabled=True),
        ]

    def test_statistics_only_different_indicators_strategy(
        self, generator: ConditionGenerator, statistics_indicators: List[IndicatorGene]
    ):
        """統計指標のみでDifferent Indicators戦略が選択された場合の診断"""
        # 統計指標のみでDIFFERENT_INDICATORS戦略を強制
        generator.set_context()  # デフォルトコンテキスト

        long_conds, short_conds, exit_conds = generator.generate_balanced_conditions(
            statistics_indicators
        )

        # 診断出力
        print(f"\n=== 統計指標のみテスト ===")
        print(f"ロング条件数: {len(long_conds)}")
        print(f"ショート条件数: {len(short_conds)}")
        print(f"ロング条件詳細: {[c.left_operand if hasattr(c, 'left_operand') else str(c) for c in long_conds]}")
        print(f"ショート条件詳細: {[c.left_operand if hasattr(c, 'left_operand') else str(c) for c in short_conds]}")

        # アサーション (これらが失敗すれば統計指標のshort条件生成に問題)
        assert len(long_conds) > 0, "統計指標でロング条件が生成されなかった"
        if len(short_conds) == 0:
            pytest.fail("統計指標でショート条件が生成されていない - DI戦略の不備を示唆")


    def test_complex_conditions_strategy_balanced(
        self, generator: ConditionGenerator, statistics_indicators: List[IndicatorGene]
    ):
        """Complex Conditions戦略でのバランス確認"""
        # Complex Conditions戦略が選ばれるケースをログで確認
        print(f"\n=== Complex Conditions戦略テスト ===")

        # 戦略タイプ選択をモックするかログで確認
        long_conds, short_conds, exit_conds = generator.generate_balanced_conditions(
            statistics_indicators
        )

        print(f"ロング条件数: {len(long_conds)}")
        print(f"ショート条件数: {len(short_conds)}")

        # Complex戦略では両方の条件が生成されるはず
        assert len(long_conds) > 0 and len(short_conds) > 0, \
            "Complex戦略では両方の条件が生成されるべき"

    def test_indicator_classification(self, generator: ConditionGenerator):
        """指標分類機能の診断"""
        test_indicators = [
            IndicatorGene(type="RSI", enabled=True),  # STATISTICS
            IndicatorGene(type="SMA", enabled=True),  # TREND
            IndicatorGene(type="ROC", enabled=True),  # MOMENTUM
        ]

        categorized = generator._dynamic_classify(test_indicators)

        print(f"\n=== 指標分類診断 ===")
        for category, indicators in categorized.items():
            print(f"{category.name}: {len(indicators)} 個")
            print(f"  - {[ind.type for ind in indicators]}")

        # 分類された指標があることを確認
        # assert categorized[IndicatorType.STATISTICS], "統計指標が分類されていない"  # 統計指標は削除済み
        assert categorized[IndicatorType.TREND], "トレンド指標が分類されていない"
        assert categorized[IndicatorType.MOMENTUM], "モメンタム指標が分類されていない"

    def test_strategy_type_selection_balance(self, generator: ConditionGenerator):
        """戦略タイプ選択のバランス診断"""
        # 異なる指標組み合わせでの戦略選択をテスト

        test_cases = [
            ("統計のみ", [IndicatorGene(type="RSI", enabled=True), IndicatorGene(type="CCI", enabled=True)]),
            ("トレンドのみ", [IndicatorGene(type="SMA", enabled=True), IndicatorGene(type="EMA", enabled=True)]),
            ("混合", [IndicatorGene(type="RSI", enabled=True), IndicatorGene(type="SMA", enabled=True), IndicatorGene(type="ROC", enabled=True)]),
            ("ML混合", [IndicatorGene(type="ML_UP_PROB", enabled=True), IndicatorGene(type="SMA", enabled=True)]),
        ]

        for case_name, indicators in test_cases:
            strategy_type = generator._select_strategy_type(indicators)
            print(f"\n=== {case_name}戦略選択 ===")
            print(f"指標: {[ind.type for ind in indicators]}")
            print(f"選択戦略: {strategy_type.name}")

            long_conds, short_conds, exit_conds = generator.generate_balanced_conditions(indicators)
            print(f"生成: ロング={len(long_conds)}, ショート={len(short_conds)}, エグジット={len(exit_conds)}")

            # 各ケースでショート条件が生成されているべき (Complex戦略以外)
            if strategy_type.name == "DIFFERENT_INDICATORS":
                assert len(short_conds) > 0, f"{case_name}でDI戦略がショート条件を生成していない"


# 直接実行用の関数
def run_diagnostics():
    """診断実行関数"""
    logging.basicConfig(level=logging.DEBUG)

    generator = ConditionGenerator(enable_smart_generation=True)

    # 診断ケース
    diagnostics = [
        ("統計指標診断", [IndicatorGene(type="RSI", enabled=True), IndicatorGene(type="STOCH", enabled=True)]),
        ("トレンド指標診断", [IndicatorGene(type="SMA", enabled=True), IndicatorGene(type="EMA", enabled=True)]),
        ("混合指標診断", [IndicatorGene(type="RSI", enabled=True), IndicatorGene(type="SMA", enabled=True), IndicatorGene(type="ROC", enabled=True)]),
    ]

    print("=" * 60)
    print("ロング・ショート戦略バランス診断結果")
    print("=" * 60)

    for name, indicators in diagnostics:
        print(f"\n{name}")
        print("-" * 40)
        print(f"指標: {[ind.type for ind in indicators]}")

        long_conds, short_conds, exit_conds = generator.generate_balanced_conditions(indicators)

        print(f"ロング条件数: {len(long_conds)}")
        print(f"ショート条件数: {len(short_conds)}")
        print(f"エグジット条件数: {len(exit_conds)}")

        if len(long_conds) > 0:
            print(f"ロング条件例: {long_conds[0].left_operand if hasattr(long_conds[0], 'left_operand') else str(long_conds[0])}")

        if len(short_conds) > 0:
            print(f"ショート条件例: {short_conds[0].left_operand if hasattr(short_conds[0], 'left_operand') else  str(short_conds[0])}")

        # 問題検出
        if len(long_conds) > 0 and len(short_conds) == 0:
            print(f"⚠️  WARNING: ショート条件が生成されていない!")
        elif len(long_conds) == 0 and len(short_conds) > 0:
            print(f"⚠️  WARNING: ロング条件が生成されていない!")
        elif len(long_conds) > 0 and len(short_conds) > 0:
            print("✅ OK: ロング・ショート条件がバランスよく生成されている")
        else:
            print("❌ ERROR: どちらの条件も生成されていない")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_diagnostics()