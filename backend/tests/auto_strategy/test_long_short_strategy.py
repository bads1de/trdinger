"""
ロング・ショート戦略のテスト

オートストラテジーシステムのロング・ショート対応機能をテストします。
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import sys
import os

# テスト用のパス設定
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "app",
        "core",
        "services",
        "auto_strategy",
        "models",
    )
)
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "app",
        "core",
        "services",
        "auto_strategy",
        "factories",
    )
)
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "app",
        "core",
        "services",
        "auto_strategy",
        "generators",
    )
)

from strategy_gene import StrategyGene, IndicatorGene, Condition
from strategy_factory import StrategyFactory
from random_gene_generator import RandomGeneGenerator


class TestLongShortStrategy(unittest.TestCase):
    """ロング・ショート戦略テストクラス"""

    def setUp(self):
        """テストセットアップ"""
        self.factory = StrategyFactory()
        self.generator = RandomGeneGenerator()

    def test_strategy_gene_long_short_fields(self):
        """StrategyGeneのロング・ショートフィールドテスト"""
        print("\n=== StrategyGeneロング・ショートフィールドテスト ===")

        # ロング・ショート条件を含む戦略遺伝子を作成
        long_conditions = [
            Condition(left_operand="RSI_14", operator="<", right_operand=30)
        ]
        short_conditions = [
            Condition(left_operand="RSI_14", operator=">", right_operand=70)
        ]

        gene = StrategyGene(
            indicators=[IndicatorGene(type="RSI", parameters={"period": 14})],
            long_entry_conditions=long_conditions,
            short_entry_conditions=short_conditions,
            exit_conditions=[
                Condition(left_operand="RSI_14", operator="==", right_operand=50)
            ],
        )

        # フィールドの存在確認
        self.assertTrue(hasattr(gene, "long_entry_conditions"))
        self.assertTrue(hasattr(gene, "short_entry_conditions"))
        self.assertEqual(len(gene.long_entry_conditions), 1)
        self.assertEqual(len(gene.short_entry_conditions), 1)

        print("✅ ロング・ショートフィールドが正しく設定されています")

    def test_effective_conditions_methods(self):
        """有効条件取得メソッドのテスト"""
        print("\n=== 有効条件取得メソッドテスト ===")

        # 新しい形式（ロング・ショート分離）
        gene_new = StrategyGene(
            long_entry_conditions=[
                Condition(left_operand="RSI_14", operator="<", right_operand=30)
            ],
            short_entry_conditions=[
                Condition(left_operand="RSI_14", operator=">", right_operand=70)
            ],
        )

        # 古い形式（後方互換性）
        gene_old = StrategyGene(
            entry_conditions=[
                Condition(left_operand="RSI_14", operator="<", right_operand=30)
            ]
        )

        # 新しい形式のテスト
        long_conds = gene_new.get_effective_long_conditions()
        short_conds = gene_new.get_effective_short_conditions()
        self.assertEqual(len(long_conds), 1)
        self.assertEqual(len(short_conds), 1)
        self.assertTrue(gene_new.has_long_short_separation())

        # 古い形式のテスト（後方互換性）
        old_long_conds = gene_old.get_effective_long_conditions()
        old_short_conds = gene_old.get_effective_short_conditions()
        self.assertEqual(len(old_long_conds), 1)  # entry_conditionsがロング条件として扱われる
        self.assertEqual(len(old_short_conds), 0)  # ショート条件はなし
        self.assertFalse(gene_old.has_long_short_separation())

        print("✅ 有効条件取得メソッドが正しく動作しています")

    def test_strategy_factory_long_short_entry(self):
        """StrategyFactoryのロング・ショートエントリーテスト"""
        print("\n=== StrategyFactoryロング・ショートエントリーテスト ===")

        # モックデータの準備
        mock_data = Mock()
        mock_data.Close = pd.Series([100, 101, 102])
        mock_data.RSI_14 = pd.Series([25, 50, 75])  # ロング→中立→ショート

        # ロング・ショート条件を含む戦略遺伝子
        gene = StrategyGene(
            indicators=[IndicatorGene(type="RSI", parameters={"period": 14})],
            long_entry_conditions=[
                Condition(left_operand="RSI_14", operator="<", right_operand=30)
            ],
            short_entry_conditions=[
                Condition(left_operand="RSI_14", operator=">", right_operand=70)
            ],
            exit_conditions=[
                Condition(left_operand="RSI_14", operator="==", right_operand=50)
            ],
        )

        # 戦略クラス生成
        strategy_class = self.factory.create_strategy_class(gene)
        self.assertIsNotNone(strategy_class)

        # 戦略インスタンス作成
        strategy_instance = strategy_class(data=mock_data, params={})
        strategy_instance.indicators = {"RSI_14": mock_data.RSI_14}

        # ロング条件チェック（RSI=25 < 30）
        long_result = strategy_instance._check_long_entry_conditions()
        self.assertTrue(long_result, "ロング条件が正しく評価されませんでした")

        # ショート条件チェック（RSI=75 > 70）
        # 最新の値を使用するため、RSI_14の最後の値（75）でテスト
        mock_data.RSI_14 = pd.Series([25, 50, 75])
        strategy_instance.indicators = {"RSI_14": mock_data.RSI_14}
        short_result = strategy_instance._check_short_entry_conditions()
        self.assertTrue(short_result, "ショート条件が正しく評価されませんでした")

        print("✅ StrategyFactoryのロング・ショートエントリーが正しく動作しています")

    def test_random_gene_generator_long_short(self):
        """RandomGeneGeneratorのロング・ショート条件生成テスト"""
        print("\n=== RandomGeneGeneratorロング・ショート条件生成テスト ===")

        # ランダム戦略遺伝子を生成
        gene = self.generator.generate_random_gene()

        # ロング・ショート条件が生成されているかチェック
        self.assertTrue(hasattr(gene, "long_entry_conditions"))
        self.assertTrue(hasattr(gene, "short_entry_conditions"))

        # 少なくとも一方の条件が存在することを確認
        has_long = len(gene.long_entry_conditions) > 0
        has_short = len(gene.short_entry_conditions) > 0
        self.assertTrue(
            has_long or has_short, "ロング・ショート条件のいずれかが生成されている必要があります"
        )

        print(f"✅ ロング条件数: {len(gene.long_entry_conditions)}")
        print(f"✅ ショート条件数: {len(gene.short_entry_conditions)}")
        print("✅ RandomGeneGeneratorがロング・ショート条件を正しく生成しています")

    def test_backward_compatibility(self):
        """後方互換性テスト"""
        print("\n=== 後方互換性テスト ===")

        # 古い形式の戦略遺伝子（entry_conditionsのみ）
        old_gene = StrategyGene(
            indicators=[IndicatorGene(type="RSI", parameters={"period": 14})],
            entry_conditions=[
                Condition(left_operand="RSI_14", operator="<", right_operand=30)
            ],
            exit_conditions=[
                Condition(left_operand="RSI_14", operator=">", right_operand=70)
            ],
        )

        # 戦略クラス生成が成功することを確認
        strategy_class = self.factory.create_strategy_class(old_gene)
        self.assertIsNotNone(strategy_class)

        # 古い形式のエントリー条件チェックメソッドが動作することを確認
        mock_data = Mock()
        mock_data.Close = pd.Series([100, 101, 102])
        mock_data.RSI_14 = pd.Series([25, 50, 75])

        strategy_instance = strategy_class(data=mock_data, params={})
        strategy_instance.indicators = {"RSI_14": mock_data.RSI_14}

        # 古いメソッドが動作することを確認
        old_entry_result = strategy_instance._check_entry_conditions()
        self.assertTrue(old_entry_result, "後方互換性のあるエントリー条件チェックが失敗しました")

        print("✅ 後方互換性が正しく保たれています")


def run_long_short_strategy_tests():
    """ロング・ショート戦略テストを実行"""
    print("🚀 ロング・ショート戦略テスト開始")

    try:
        # テストスイートを作成
        suite = unittest.TestLoader().loadTestsFromTestCase(TestLongShortStrategy)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        if result.wasSuccessful():
            print("\n🎉 全てのロング・ショート戦略テストが成功しました！")
            return True
        else:
            print(f"\n❌ テスト失敗: {len(result.failures)} failures, {len(result.errors)} errors")
            return False

    except Exception as e:
        print(f"\n❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_long_short_strategy_tests()
