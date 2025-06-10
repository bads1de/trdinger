"""
OI/FR対応StrategyFactoryのテスト

StrategyFactoryのOI/FR条件処理機能をテストします。
"""

import unittest
from unittest.mock import Mock, MagicMock
import pandas as pd
import numpy as np
import logging

# テスト対象のインポート
from backend.app.core.services.auto_strategy.models.strategy_gene import (
    StrategyGene,
    IndicatorGene,
    Condition,
)
from backend.app.core.services.auto_strategy.factories.strategy_factory import (
    StrategyFactory,
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestOIFRStrategyFactory(unittest.TestCase):
    """OI/FR対応StrategyFactoryのテストクラス"""

    def setUp(self):
        """テストセットアップ"""
        self.factory = StrategyFactory()

    def test_oi_fr_condition_validation(self):
        """OI/FR条件の妥当性検証テスト"""
        print("\n=== OI/FR条件妥当性検証テスト ===")

        # OI/FR条件を含む戦略遺伝子を作成
        gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            ],
            entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="SMA_20"),
                Condition(
                    left_operand="FundingRate", operator=">", right_operand=0.0005
                ),
                Condition(
                    left_operand="OpenInterest", operator=">", right_operand=10000000
                ),
            ],
            exit_conditions=[
                Condition(left_operand="RSI_14", operator=">", right_operand=70),
                Condition(
                    left_operand="FundingRate", operator="<", right_operand=-0.0005
                ),
            ],
        )

        # 妥当性検証
        is_valid, errors = self.factory.validate_gene(gene)

        print(f"妥当性: {is_valid}")
        if errors:
            print(f"エラー: {errors}")

        # OI/FR条件は有効なデータソースとして認識されるべき
        self.assertTrue(is_valid, f"OI/FR条件が無効と判定されました: {errors}")
        print("✅ OI/FR条件の妥当性検証成功")

    def test_strategy_class_generation_with_oi_fr(self):
        """OI/FR条件を含む戦略クラス生成テスト"""
        print("\n=== OI/FR戦略クラス生成テスト ===")

        # OI/FR条件を含む戦略遺伝子
        gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            ],
            entry_conditions=[
                Condition(
                    left_operand="FundingRate", operator=">", right_operand=0.001
                ),
            ],
            exit_conditions=[
                Condition(
                    left_operand="OpenInterest", operator="<", right_operand=5000000
                ),
            ],
        )

        # 戦略クラス生成
        try:
            strategy_class = self.factory.create_strategy_class(gene)
            self.assertIsNotNone(strategy_class)
            print("✅ OI/FR戦略クラス生成成功")

            # クラス名確認
            expected_name = f"GeneratedStrategy_{gene.id}"
            self.assertEqual(strategy_class.__name__, expected_name)
            print(f"✅ クラス名確認: {strategy_class.__name__}")

        except Exception as e:
            self.fail(f"OI/FR戦略クラス生成に失敗: {e}")

    def test_oi_fr_data_access_simulation(self):
        """OI/FRデータアクセスのシミュレーションテスト"""
        print("\n=== OI/FRデータアクセスシミュレーション ===")

        # モックデータの作成
        mock_data = Mock()
        mock_data.Close = pd.Series([100, 101, 102, 103, 104])
        mock_data.High = pd.Series([101, 102, 103, 104, 105])
        mock_data.Low = pd.Series([99, 100, 101, 102, 103])
        mock_data.Open = pd.Series([100, 101, 102, 103, 104])
        mock_data.Volume = pd.Series([1000, 1100, 1200, 1300, 1400])

        # OI/FRデータを追加
        mock_data.OpenInterest = pd.Series(
            [10000000, 11000000, 12000000, 13000000, 14000000]
        )
        mock_data.FundingRate = pd.Series([0.0001, 0.0003, 0.0005, 0.0007, 0.0009])

        # 戦略遺伝子
        gene = StrategyGene(
            indicators=[],
            entry_conditions=[
                Condition(
                    left_operand="FundingRate", operator=">", right_operand=0.0005
                ),
            ],
            exit_conditions=[
                Condition(
                    left_operand="OpenInterest", operator=">", right_operand=12000000
                ),
            ],
        )

        # 戦略クラス生成
        strategy_class = self.factory.create_strategy_class(gene)

        # 戦略インスタンス作成（モックデータ使用）
        strategy_instance = strategy_class(data=mock_data, params={})

        # _get_oi_fr_value メソッドのテスト
        fr_value = strategy_instance._get_oi_fr_value("FundingRate")
        oi_value = strategy_instance._get_oi_fr_value("OpenInterest")

        print(f"FundingRate値: {fr_value}")
        print(f"OpenInterest値: {oi_value}")

        # 最新の値が取得されることを確認
        self.assertEqual(fr_value, 0.0009)  # 最後の値
        self.assertEqual(oi_value, 14000000)  # 最後の値

        print("✅ OI/FRデータアクセス成功")

    def test_condition_evaluation_with_oi_fr(self):
        """OI/FR条件評価テスト"""
        print("\n=== OI/FR条件評価テスト ===")

        # モックデータの準備
        mock_data = Mock()
        mock_data.Close = pd.Series([100, 101, 102])
        mock_data.OpenInterest = pd.Series(
            [8000000, 9000000, 15000000]
        )  # 最後が閾値超え
        mock_data.FundingRate = pd.Series([0.0001, 0.0003, 0.0008])  # 最後が閾値超え

        # 戦略遺伝子
        gene = StrategyGene(
            indicators=[],
            entry_conditions=[
                Condition(
                    left_operand="FundingRate", operator=">", right_operand=0.0005
                ),
                Condition(
                    left_operand="OpenInterest", operator=">", right_operand=10000000
                ),
            ],
            exit_conditions=[
                Condition(
                    left_operand="close", operator="<", right_operand=95
                ),  # 基本的なイグジット条件
            ],
        )

        # 戦略クラス生成とインスタンス作成
        strategy_class = self.factory.create_strategy_class(gene)
        strategy_instance = strategy_class(data=mock_data, params={})
        strategy_instance.indicators = {}

        # 条件評価テスト
        fr_condition = gene.entry_conditions[0]  # FundingRate > 0.0005
        oi_condition = gene.entry_conditions[1]  # OpenInterest > 10000000

        fr_result = strategy_instance._evaluate_condition(fr_condition)
        oi_result = strategy_instance._evaluate_condition(oi_condition)

        print(f"FundingRate条件評価: {fr_result} (0.0008 > 0.0005)")
        print(f"OpenInterest条件評価: {oi_result} (15000000 > 10000000)")

        # 両方ともTrueになるはず
        self.assertTrue(fr_result, "FundingRate条件が正しく評価されませんでした")
        self.assertTrue(oi_result, "OpenInterest条件が正しく評価されませんでした")

        print("✅ OI/FR条件評価成功")

    def test_invalid_oi_fr_data_handling(self):
        """無効なOI/FRデータの処理テスト"""
        print("\n=== 無効OI/FRデータ処理テスト ===")

        # OI/FRデータが存在しないモックデータ
        mock_data = Mock()
        mock_data.Close = pd.Series([100, 101, 102])
        # OpenInterestとFundingRateは存在しない

        gene = StrategyGene(
            indicators=[],
            entry_conditions=[
                Condition(
                    left_operand="FundingRate", operator=">", right_operand=0.0005
                ),
            ],
            exit_conditions=[
                Condition(
                    left_operand="close", operator="<", right_operand=95
                ),  # 基本的なイグジット条件
            ],
        )

        strategy_class = self.factory.create_strategy_class(gene)
        strategy_instance = strategy_class(data=mock_data, params={})
        strategy_instance.indicators = {}

        # 存在しないデータへのアクセステスト
        fr_value = strategy_instance._get_oi_fr_value("FundingRate")
        oi_value = strategy_instance._get_oi_fr_value("OpenInterest")

        print(f"存在しないFundingRate値: {fr_value}")
        print(f"存在しないOpenInterest値: {oi_value}")

        # デフォルト値（0.0）が返されることを確認
        self.assertEqual(fr_value, 0.0)
        self.assertEqual(oi_value, 0.0)

        print("✅ 無効OI/FRデータ処理成功")


def run_oi_fr_strategy_factory_tests():
    """OI/FR StrategyFactoryテストの実行"""
    print("🧪 OI/FR対応StrategyFactoryテスト開始")

    # テストスイート作成
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOIFRStrategyFactory)

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 結果サマリー
    print(f"\n📊 テスト結果サマリー:")
    print(f"  実行テスト数: {result.testsRun}")
    print(f"  失敗: {len(result.failures)}")
    print(f"  エラー: {len(result.errors)}")

    if result.failures:
        print("❌ 失敗したテスト:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")

    if result.errors:
        print("💥 エラーが発生したテスト:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print("🎉 全てのOI/FR StrategyFactoryテストが成功しました！")
    else:
        print("⚠️ 一部のテストが失敗しました。")

    return success


if __name__ == "__main__":
    run_oi_fr_strategy_factory_tests()
