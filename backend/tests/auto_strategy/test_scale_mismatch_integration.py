"""
スケール不一致問題修正の統合テスト

修正されたオートストラテジーシステム全体の動作を検証し、
取引回数0問題が解決されていることを確認します。
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from app.core.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.core.services.auto_strategy.engines.fitness_calculator import FitnessCalculator
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.core.services.auto_strategy.models.strategy_gene import (
    StrategyGene,
    IndicatorGene,
    Condition,
)
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.utils.operand_grouping import (
    operand_grouping_system,
)


class TestScaleMismatchIntegration:
    """スケール不一致問題修正の統合テスト"""

    def setup_method(self):
        """テストセットアップ"""
        config = {
            "min_indicators": 2,
            "max_indicators": 3,
            "min_conditions": 1,
            "max_conditions": 2,
        }
        self.generator = RandomGeneGenerator(config)

        # モックのバックテストサービス
        self.mock_backtest_service = Mock()

        # フィットネス計算器
        self.fitness_calculator = FitnessCalculator(
            backtest_service=self.mock_backtest_service, strategy_factory=Mock()
        )

        # GA設定
        self.ga_config = GAConfig(
            population_size=10, generations=5, crossover_rate=0.8, mutation_rate=0.2
        )

    def test_improved_condition_generation_quality(self):
        """改善された条件生成の品質をテスト"""
        # 大量の戦略を生成して統計を取る
        strategies = []
        for _ in range(50):
            strategy = self.generator.generate_random_gene()
            strategies.append(strategy)

        # 生成された条件の分析
        total_conditions = 0
        scale_compatible_conditions = 0
        numerical_conditions = 0

        for strategy in strategies:
            all_conditions = strategy.entry_conditions + strategy.exit_conditions

            for condition in all_conditions:
                total_conditions += 1

                if isinstance(condition.right_operand, (int, float)):
                    numerical_conditions += 1
                elif isinstance(condition.right_operand, str):
                    compatibility = operand_grouping_system.get_compatibility_score(
                        condition.left_operand, condition.right_operand
                    )
                    if compatibility >= 0.3:  # 中程度以上の互換性
                        scale_compatible_conditions += 1

        # 品質指標の計算
        numerical_ratio = (
            numerical_conditions / total_conditions if total_conditions > 0 else 0
        )
        compatibility_ratio = (
            scale_compatible_conditions / (total_conditions - numerical_conditions)
            if (total_conditions - numerical_conditions) > 0
            else 0
        )

        print(f"\n=== 条件生成品質分析 ===")
        print(f"総条件数: {total_conditions}")
        print(f"数値比較: {numerical_conditions} ({numerical_ratio:.1%})")
        print(
            f"互換性のある指標比較: {scale_compatible_conditions} ({compatibility_ratio:.1%})"
        )

        # 品質基準の確認
        assert (
            numerical_ratio >= 0.3
        ), f"数値比較の割合が低すぎます: {numerical_ratio:.1%}"
        assert (
            compatibility_ratio >= 0.7
        ), f"互換性のある比較の割合が低すぎます: {compatibility_ratio:.1%}"

    def test_data_coverage_penalty_integration(self):
        """データカバレッジペナルティの統合テスト"""
        # OI/FRを使用する戦略
        strategy_with_oi_fr = StrategyGene(
            indicators=[
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)
            ],
            entry_conditions=[
                Condition("OpenInterest", ">", 1000000),
                Condition("RSI", "<", 30),
            ],
            exit_conditions=[Condition("FundingRate", ">", 0.001)],
        )

        # OI/FRを使用しない戦略
        strategy_without_oi_fr = StrategyGene(
            indicators=[
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            ],
            entry_conditions=[Condition("RSI", "<", 30)],
            exit_conditions=[Condition("close", ">", "SMA")],
        )

        # テストデータ（OI/FRが部分的に欠損）
        test_data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104],
                "OpenInterest": [1000000, 1100000, 0.0, 0.0, 1200000],  # 40%欠損
                "FundingRate": [0.001, 0.002, 0.0, 0.001, 0.002],  # 20%欠損
            }
        )

        # モックのバックテスト結果
        mock_result_with_oi_fr = {
            "performance_metrics": {
                "total_return": 15.0,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
                "total_trades": 10,
            },
            "data": test_data,
        }

        mock_result_without_oi_fr = {
            "performance_metrics": {
                "total_return": 15.0,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
                "total_trades": 10,
            },
            "data": test_data,
        }

        # フィットネス計算
        fitness_with_oi_fr = self.fitness_calculator.calculate_fitness(
            mock_result_with_oi_fr, self.ga_config, strategy_with_oi_fr
        )

        fitness_without_oi_fr = self.fitness_calculator.calculate_fitness(
            mock_result_without_oi_fr, self.ga_config, strategy_without_oi_fr
        )

        print(f"\n=== データカバレッジペナルティテスト ===")
        print(f"OI/FR使用戦略のフィットネス: {fitness_with_oi_fr:.4f}")
        print(f"OI/FR未使用戦略のフィットネス: {fitness_without_oi_fr:.4f}")

        # OI/FRを使用する戦略のフィットネスが低下していることを確認
        assert (
            fitness_with_oi_fr < fitness_without_oi_fr
        ), "データカバレッジペナルティが適用されていません"

    def test_realistic_strategy_generation(self):
        """現実的な戦略生成のテスト"""
        # 複数の戦略を生成して、現実的な条件が生成されることを確認
        realistic_strategies = 0
        total_strategies = 20

        for _ in range(total_strategies):
            strategy = self.generator.generate_random_gene()

            # 現実的な戦略の基準
            is_realistic = True

            # すべての条件をチェック
            all_conditions = strategy.entry_conditions + strategy.exit_conditions

            for condition in all_conditions:
                # 数値比較の場合、適切な範囲内かチェック
                if isinstance(condition.right_operand, (int, float)):
                    left_group = operand_grouping_system.get_operand_group(
                        condition.left_operand
                    )

                    # RSIなどの0-100指標が適切な範囲内かチェック
                    if left_group.value == "percentage_0_100":
                        if not (0 <= condition.right_operand <= 100):
                            is_realistic = False
                            break

                    # FundingRateが適切な範囲内かチェック
                    elif "FundingRate" in condition.left_operand:
                        if not (-0.01 <= condition.right_operand <= 0.01):
                            is_realistic = False
                            break

                # 指標比較の場合、互換性をチェック
                elif isinstance(condition.right_operand, str):
                    compatibility = operand_grouping_system.get_compatibility_score(
                        condition.left_operand, condition.right_operand
                    )
                    if compatibility < 0.1:  # 非常に低い互換性は非現実的
                        is_realistic = False
                        break

            if is_realistic:
                realistic_strategies += 1

        realistic_ratio = realistic_strategies / total_strategies
        print(f"\n=== 現実的な戦略生成テスト ===")
        print(f"現実的な戦略の割合: {realistic_ratio:.1%}")

        # 80%以上が現実的な戦略であることを確認
        assert (
            realistic_ratio >= 0.8
        ), f"現実的な戦略の割合が低すぎます: {realistic_ratio:.1%}"

    def test_condition_diversity(self):
        """条件の多様性テスト"""
        # 多様な条件が生成されることを確認
        left_operands = set()
        right_operands = set()
        operators = set()

        for _ in range(30):
            strategy = self.generator.generate_random_gene()
            all_conditions = strategy.entry_conditions + strategy.exit_conditions

            for condition in all_conditions:
                left_operands.add(condition.left_operand)
                operators.add(condition.operator)

                if isinstance(condition.right_operand, str):
                    right_operands.add(condition.right_operand)

        print(f"\n=== 条件多様性テスト ===")
        print(f"左オペランドの種類: {len(left_operands)}")
        print(f"右オペランドの種類: {len(right_operands)}")
        print(f"演算子の種類: {len(operators)}")
        print(f"左オペランド: {sorted(left_operands)}")
        print(f"右オペランド: {sorted(right_operands)}")

        # 十分な多様性があることを確認
        assert len(left_operands) >= 5, "左オペランドの多様性が不足しています"
        assert len(right_operands) >= 3, "右オペランドの多様性が不足しています"
        assert len(operators) >= 2, "演算子の多様性が不足しています"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
