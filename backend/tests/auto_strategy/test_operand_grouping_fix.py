"""
オペランドグループ化システムとスケール不一致修正のテスト

修正されたrandom_gene_generatorとoperand_groupingシステムの動作を検証します。
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from collections import Counter

from app.core.services.auto_strategy.utils.operand_grouping import (
    operand_grouping_system,
    OperandGroup,
)
from app.core.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.core.services.auto_strategy.models.strategy_gene import (
    StrategyGene,
    IndicatorGene,
    Condition,
)
from app.core.services.auto_strategy.utils.data_coverage_analyzer import (
    data_coverage_analyzer,
)


class TestOperandGroupingSystem:
    """オペランドグループ化システムのテスト"""

    def test_operand_classification(self):
        """オペランドの分類が正しく行われることを確認"""
        # 価格ベース
        assert (
            operand_grouping_system.get_operand_group("close")
            == OperandGroup.PRICE_BASED
        )
        assert (
            operand_grouping_system.get_operand_group("SMA") == OperandGroup.PRICE_BASED
        )
        assert (
            operand_grouping_system.get_operand_group("EMA") == OperandGroup.PRICE_BASED
        )

        # 0-100%オシレーター
        assert (
            operand_grouping_system.get_operand_group("RSI")
            == OperandGroup.PERCENTAGE_0_100
        )
        assert (
            operand_grouping_system.get_operand_group("STOCH")
            == OperandGroup.PERCENTAGE_0_100
        )
        assert (
            operand_grouping_system.get_operand_group("ADX")
            == OperandGroup.PERCENTAGE_0_100
        )

        # ±100オシレーター
        assert (
            operand_grouping_system.get_operand_group("CCI")
            == OperandGroup.PERCENTAGE_NEG100_100
        )

        # ゼロ中心
        assert (
            operand_grouping_system.get_operand_group("MACD")
            == OperandGroup.ZERO_CENTERED
        )
        assert (
            operand_grouping_system.get_operand_group("OBV")
            == OperandGroup.ZERO_CENTERED
        )

        # 特殊スケール
        assert (
            operand_grouping_system.get_operand_group("OpenInterest")
            == OperandGroup.SPECIAL_SCALE
        )
        assert (
            operand_grouping_system.get_operand_group("FundingRate")
            == OperandGroup.SPECIAL_SCALE
        )

    def test_compatibility_scores(self):
        """互換性スコアが適切に計算されることを確認"""
        # 同一グループ内は高い互換性
        score = operand_grouping_system.get_compatibility_score("RSI", "STOCH")
        assert score >= 0.8

        score = operand_grouping_system.get_compatibility_score("close", "SMA")
        assert score >= 0.8

        # 異なるグループ間は低い互換性
        score = operand_grouping_system.get_compatibility_score("close", "RSI")
        assert score <= 0.3

        score = operand_grouping_system.get_compatibility_score(
            "OpenInterest", "FundingRate"
        )
        assert score <= 0.5  # 特殊スケール同士でも中程度

    def test_compatible_operands_retrieval(self):
        """互換性の高いオペランドの取得が正しく動作することを確認"""
        available_operands = ["close", "RSI", "SMA", "STOCH", "OpenInterest"]

        # RSIと互換性の高いオペランド
        compatible = operand_grouping_system.get_compatible_operands(
            "RSI", available_operands, min_compatibility=0.8
        )
        assert "STOCH" in compatible
        assert "close" not in compatible
        assert "SMA" not in compatible

        # closeと互換性の高いオペランド
        compatible = operand_grouping_system.get_compatible_operands(
            "close", available_operands, min_compatibility=0.8
        )
        assert "SMA" in compatible
        assert "RSI" not in compatible


class TestRandomGeneGeneratorFix:
    """修正されたRandomGeneGeneratorのテスト"""

    def setup_method(self):
        """テストセットアップ"""
        config = {
            "min_indicators": 2,
            "max_indicators": 3,
            "min_conditions": 1,
            "max_conditions": 2,
        }
        self.generator = RandomGeneGenerator(config)

    def test_improved_operand_selection(self):
        """改善されたオペランド選択ロジックのテスト"""
        # テスト用の指標リスト
        indicators = [
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            IndicatorGene(type="CCI", parameters={"period": 20}, enabled=True),
        ]

        # 複数回条件生成を実行して統計を取る
        condition_pairs = []
        for _ in range(100):
            condition = self.generator._generate_single_condition(indicators, "entry")
            if isinstance(condition.right_operand, str):  # 数値でない場合のみ
                condition_pairs.append(
                    (condition.left_operand, condition.right_operand)
                )

        # スケール不一致の条件が減少していることを確認
        scale_mismatches = 0
        for left, right in condition_pairs:
            compatibility = operand_grouping_system.get_compatibility_score(left, right)
            if compatibility <= 0.3:
                scale_mismatches += 1

        # スケール不一致の割合が大幅に減少していることを確認（従来は70%程度）
        mismatch_ratio = (
            scale_mismatches / len(condition_pairs) if condition_pairs else 0
        )
        assert (
            mismatch_ratio < 0.3
        ), f"スケール不一致の割合が高すぎます: {mismatch_ratio:.2%}"

    def test_compatible_operand_selection(self):
        """互換性の高いオペランド選択のテスト"""
        indicators = [
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            IndicatorGene(type="STOCH", parameters={"period": 14}, enabled=True),
        ]

        # RSIを左オペランドとして、互換性の高い右オペランドが選択されることを確認
        compatible_selections = 0
        total_selections = 100

        for _ in range(total_selections):
            right_operand = self.generator._choose_compatible_operand("RSI", indicators)
            if isinstance(right_operand, str):
                compatibility = operand_grouping_system.get_compatibility_score(
                    "RSI", right_operand
                )
                if compatibility >= 0.3:  # 中程度以上の互換性
                    compatible_selections += 1

        # 80%以上が互換性の高い選択であることを確認
        compatibility_ratio = compatible_selections / total_selections
        assert (
            compatibility_ratio >= 0.8
        ), f"互換性の高い選択の割合が低すぎます: {compatibility_ratio:.2%}"

    def test_numerical_comparison_increase(self):
        """数値比較の増加を確認"""
        indicators = [
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
        ]

        numerical_comparisons = 0
        total_conditions = 100

        for _ in range(total_conditions):
            condition = self.generator._generate_single_condition(indicators, "entry")
            if isinstance(condition.right_operand, (int, float)):
                numerical_comparisons += 1

        # 数値比較の割合が40%程度であることを確認（従来は30%）
        numerical_ratio = numerical_comparisons / total_conditions
        assert (
            0.3 <= numerical_ratio <= 0.5
        ), f"数値比較の割合が期待範囲外です: {numerical_ratio:.2%}"


class TestDataCoverageAnalyzer:
    """データカバレッジ分析システムのテスト"""

    def test_special_data_source_extraction(self):
        """特殊データソースの抽出が正しく動作することを確認"""
        # OI/FRを使用する戦略
        strategy_with_special = StrategyGene(
            indicators=[
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)
            ],
            entry_conditions=[
                Condition("OpenInterest", ">", 1000000),
                Condition("RSI", "<", 30),
            ],
            exit_conditions=[Condition("FundingRate", ">", 0.001)],
        )

        # テストデータ（OI/FRが部分的に欠損）
        test_data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104],
                "OpenInterest": [1000000, 1100000, 0.0, 0.0, 1200000],  # 40%欠損
                "FundingRate": [0.001, 0.002, 0.0, 0.001, 0.002],  # 20%欠損
            }
        )

        analysis = data_coverage_analyzer.analyze_strategy_coverage(
            strategy_with_special, test_data
        )

        assert analysis["uses_special_data"] is True
        assert "OpenInterest" in analysis["used_special_sources"]
        assert "FundingRate" in analysis["used_special_sources"]
        assert analysis["coverage_penalty"] > 0.0
        assert analysis["overall_coverage_score"] < 1.0

    def test_no_special_data_strategy(self):
        """特殊データソースを使用しない戦略のテスト"""
        strategy_normal = StrategyGene(
            indicators=[
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            ],
            entry_conditions=[Condition("RSI", "<", 30)],
            exit_conditions=[Condition("close", ">", "SMA")],
        )

        test_data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104],
                "OpenInterest": [0.0, 0.0, 0.0, 0.0, 0.0],  # 完全欠損
                "FundingRate": [0.0, 0.0, 0.0, 0.0, 0.0],  # 完全欠損
            }
        )

        analysis = data_coverage_analyzer.analyze_strategy_coverage(
            strategy_normal, test_data
        )

        assert analysis["uses_special_data"] is False
        assert analysis["coverage_penalty"] == 0.0
        assert analysis["overall_coverage_score"] == 1.0

    def test_coverage_penalty_calculation(self):
        """カバレッジペナルティの計算が正しく動作することを確認"""
        # 異なるカバレッジレベルでのペナルティをテスト
        test_cases = [
            (1.0, 0.0),  # 100%カバレッジ -> ペナルティなし
            (0.95, 0.0),  # 95%カバレッジ -> ペナルティなし
            (0.85, 0.1),  # 85%カバレッジ -> 軽微なペナルティ
            (0.65, 0.25),  # 65%カバレッジ -> 中程度のペナルティ
            (0.45, 0.5),  # 45%カバレッジ -> 重いペナルティ
            (0.20, 0.8),  # 20%カバレッジ -> 非常に重いペナルティ
        ]

        for coverage_ratio, expected_penalty in test_cases:
            penalty = data_coverage_analyzer._calculate_coverage_penalty(coverage_ratio)
            assert (
                penalty == expected_penalty
            ), f"カバレッジ{coverage_ratio:.0%}のペナルティが期待値と異なります"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
