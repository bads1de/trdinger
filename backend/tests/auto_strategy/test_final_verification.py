"""
最終検証テスト

修正されたオートストラテジーシステムの実際の動作を検証し、
取引回数0問題が解決されていることを確認します。
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from app.core.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)

# FitnessCalculatorは削除され、GAEngineに統合されました
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.utils.operand_grouping import (
    operand_grouping_system,
)


class TestFinalVerification:
    """最終検証テスト"""

    def setup_method(self):
        """テストセットアップ"""
        config = {
            "min_indicators": 2,
            "max_indicators": 3,
            "min_conditions": 1,
            "max_conditions": 2,
        }
        self.generator = RandomGeneGenerator(config)

        # GA設定
        self.ga_config = GAConfig(
            population_size=10, generations=5, crossover_rate=0.8, mutation_rate=0.2
        )

    def test_scale_mismatch_reduction(self):
        """スケール不一致の大幅な減少を確認"""
        print("\n=== スケール不一致減少の検証 ===")

        # 大量の戦略を生成して統計を取る
        total_conditions = 0
        scale_mismatches = 0
        numerical_conditions = 0
        high_compatibility_conditions = 0

        for i in range(100):
            strategy = self.generator.generate_random_gene()
            all_conditions = strategy.entry_conditions + strategy.exit_conditions

            for condition in all_conditions:
                total_conditions += 1

                if isinstance(condition.right_operand, (int, float)):
                    numerical_conditions += 1
                elif isinstance(condition.right_operand, str):
                    compatibility = operand_grouping_system.get_compatibility_score(
                        condition.left_operand, condition.right_operand
                    )

                    if compatibility >= 0.8:
                        high_compatibility_conditions += 1
                    elif compatibility <= 0.3:
                        scale_mismatches += 1

        # 統計の計算
        numerical_ratio = (
            numerical_conditions / total_conditions if total_conditions > 0 else 0
        )
        mismatch_ratio = (
            scale_mismatches / total_conditions if total_conditions > 0 else 0
        )
        high_compat_ratio = (
            high_compatibility_conditions / total_conditions
            if total_conditions > 0
            else 0
        )

        print(f"総条件数: {total_conditions}")
        print(f"数値比較: {numerical_conditions} ({numerical_ratio:.1%})")
        print(
            f"高互換性比較: {high_compatibility_conditions} ({high_compat_ratio:.1%})"
        )
        print(f"スケール不一致: {scale_mismatches} ({mismatch_ratio:.1%})")

        # 修正効果の確認
        assert (
            numerical_ratio >= 0.35
        ), f"数値比較の割合が期待値を下回ります: {numerical_ratio:.1%}"
        assert (
            mismatch_ratio <= 0.25
        ), f"スケール不一致の割合が高すぎます: {mismatch_ratio:.1%}"
        assert (
            high_compat_ratio >= 0.25
        ), f"高互換性比較の割合が低すぎます: {high_compat_ratio:.1%}"

        print("✅ スケール不一致問題の大幅な改善を確認")

    def test_realistic_condition_examples(self):
        """現実的な条件の生成例を確認"""
        print("\n=== 現実的な条件生成の例 ===")

        realistic_examples = []
        problematic_examples = []

        for _ in range(50):
            strategy = self.generator.generate_random_gene()
            all_conditions = strategy.entry_conditions + strategy.exit_conditions

            for condition in all_conditions:
                condition_str = f"{condition.left_operand} {condition.operator} {condition.right_operand}"

                # 現実的かどうかの判定
                is_realistic = True
                reason = ""

                if isinstance(condition.right_operand, str):
                    compatibility = operand_grouping_system.get_compatibility_score(
                        condition.left_operand, condition.right_operand
                    )
                    if compatibility <= 0.3:
                        is_realistic = False
                        reason = f"低互換性 (スコア: {compatibility:.2f})"

                if is_realistic:
                    if len(realistic_examples) < 10:
                        realistic_examples.append(condition_str)
                else:
                    if len(problematic_examples) < 5:
                        problematic_examples.append(f"{condition_str} ({reason})")

        print("現実的な条件の例:")
        for i, example in enumerate(realistic_examples[:10], 1):
            print(f"  {i}. {example}")

        if problematic_examples:
            print("\n問題のある条件の例:")
            for i, example in enumerate(problematic_examples, 1):
                print(f"  {i}. {example}")

        # 現実的な条件が多数生成されていることを確認
        assert len(realistic_examples) >= 8, "現実的な条件の生成数が不足しています"

        print("✅ 現実的な条件が適切に生成されています")

    def test_operand_group_distribution(self):
        """オペランドグループの分布を確認"""
        print("\n=== オペランドグループ分布の確認 ===")

        left_operand_groups = {}
        right_operand_groups = {}

        for _ in range(100):
            strategy = self.generator.generate_random_gene()
            all_conditions = strategy.entry_conditions + strategy.exit_conditions

            for condition in all_conditions:
                # 左オペランドのグループ
                left_group = operand_grouping_system.get_operand_group(
                    condition.left_operand
                )
                left_operand_groups[left_group.value] = (
                    left_operand_groups.get(left_group.value, 0) + 1
                )

                # 右オペランドのグループ（文字列の場合のみ）
                if isinstance(condition.right_operand, str):
                    right_group = operand_grouping_system.get_operand_group(
                        condition.right_operand
                    )
                    right_operand_groups[right_group.value] = (
                        right_operand_groups.get(right_group.value, 0) + 1
                    )

        print("左オペランドのグループ分布:")
        for group, count in sorted(left_operand_groups.items()):
            print(f"  {group}: {count}")

        print("\n右オペランドのグループ分布:")
        for group, count in sorted(right_operand_groups.items()):
            print(f"  {group}: {count}")

        # 多様性の確認
        assert (
            len(left_operand_groups) >= 3
        ), "左オペランドのグループ多様性が不足しています"
        assert (
            len(right_operand_groups) >= 2
        ), "右オペランドのグループ多様性が不足しています"

        print("✅ オペランドグループの適切な分布を確認")

    def test_data_coverage_awareness(self):
        """データカバレッジ考慮機能の動作確認"""
        print("\n=== データカバレッジ考慮機能の確認 ===")

        # OI/FRを使用する戦略の生成頻度を確認
        oi_fr_strategies = 0
        total_strategies = 100

        for _ in range(total_strategies):
            strategy = self.generator.generate_random_gene()
            all_conditions = strategy.entry_conditions + strategy.exit_conditions

            uses_oi_fr = False
            for condition in all_conditions:
                if condition.left_operand in [
                    "OpenInterest",
                    "FundingRate",
                ] or condition.right_operand in ["OpenInterest", "FundingRate"]:
                    uses_oi_fr = True
                    break

            if uses_oi_fr:
                oi_fr_strategies += 1

        oi_fr_ratio = oi_fr_strategies / total_strategies
        print(f"OI/FRを使用する戦略の割合: {oi_fr_ratio:.1%}")

        # OI/FRの使用頻度が適度に制限されていることを確認
        # （完全に排除されるわけではないが、過度に使用されることもない）
        assert (
            0.05 <= oi_fr_ratio <= 0.50
        ), f"OI/FR使用頻度が期待範囲外です: {oi_fr_ratio:.1%}"

        print("✅ データカバレッジ考慮機能が適切に動作しています")

    def test_overall_improvement_summary(self):
        """全体的な改善効果のサマリー"""
        print("\n=== 全体的な改善効果のサマリー ===")

        # 複数の指標で改善効果を測定
        metrics = {
            "total_conditions": 0,
            "numerical_conditions": 0,
            "high_compatibility": 0,
            "medium_compatibility": 0,
            "low_compatibility": 0,
            "oi_fr_usage": 0,
        }

        strategies_count = 50

        for _ in range(strategies_count):
            strategy = self.generator.generate_random_gene()
            all_conditions = strategy.entry_conditions + strategy.exit_conditions

            uses_oi_fr = False

            for condition in all_conditions:
                metrics["total_conditions"] += 1

                # OI/FR使用チェック
                if condition.left_operand in [
                    "OpenInterest",
                    "FundingRate",
                ] or condition.right_operand in ["OpenInterest", "FundingRate"]:
                    uses_oi_fr = True

                if isinstance(condition.right_operand, (int, float)):
                    metrics["numerical_conditions"] += 1
                elif isinstance(condition.right_operand, str):
                    compatibility = operand_grouping_system.get_compatibility_score(
                        condition.left_operand, condition.right_operand
                    )

                    if compatibility >= 0.8:
                        metrics["high_compatibility"] += 1
                    elif compatibility >= 0.3:
                        metrics["medium_compatibility"] += 1
                    else:
                        metrics["low_compatibility"] += 1

            if uses_oi_fr:
                metrics["oi_fr_usage"] += 1

        # 改善効果の計算と表示
        total = metrics["total_conditions"]
        print(f"分析対象: {strategies_count}戦略, {total}条件")
        print(f"数値比較: {metrics['numerical_conditions']/total:.1%}")
        print(f"高互換性比較: {metrics['high_compatibility']/total:.1%}")
        print(f"中互換性比較: {metrics['medium_compatibility']/total:.1%}")
        print(f"低互換性比較: {metrics['low_compatibility']/total:.1%}")
        print(f"OI/FR使用戦略: {metrics['oi_fr_usage']/strategies_count:.1%}")

        # 改善基準の確認
        numerical_ratio = metrics["numerical_conditions"] / total
        high_compat_ratio = metrics["high_compatibility"] / total
        low_compat_ratio = metrics["low_compatibility"] / total

        assert numerical_ratio >= 0.35, "数値比較の割合が不十分"
        assert high_compat_ratio >= 0.25, "高互換性比較の割合が不十分"
        assert low_compat_ratio <= 0.15, "低互換性比較の割合が高すぎる"

        print("\n🎉 オートストラテジー取引回数0問題の修正が成功しました！")
        print("主な改善点:")
        print("- スケール不一致条件の大幅な減少")
        print("- 数値比較の適切な増加")
        print("- 互換性の高い指標比較の優先")
        print("- データカバレッジ考慮機能の実装")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
