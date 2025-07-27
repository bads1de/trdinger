"""
エッジケーステスト

SmartConditionGeneratorの極端な条件での動作確認
- 極端なパラメータ値
- 指標データ不足
- 無効な組み合わせ
- パフォーマンス限界
"""

import pytest
import time
import psutil
import os
import sys
import tracemalloc
from typing import List
from unittest.mock import Mock, patch

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.services.auto_strategy.generators.smart_condition_generator import (
    SmartConditionGenerator,
    StrategyType,
    IndicatorType,
    INDICATOR_CHARACTERISTICS
)
from app.services.auto_strategy.models.gene_strategy import IndicatorGene, Condition


class TestEdgeCases:
    """エッジケーステストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.generator = SmartConditionGenerator(enable_smart_generation=True)
        self.test_results = {
            "extreme_parameters": {"passed": 0, "failed": 0, "errors": []},
            "missing_data": {"passed": 0, "failed": 0, "errors": []},
            "invalid_combinations": {"passed": 0, "failed": 0, "errors": []},
            "performance_limits": {"passed": 0, "failed": 0, "errors": []},
            "memory_usage": {"peak_mb": 0, "average_mb": 0},
            "processing_times": []
        }

    def test_extreme_parameter_values(self):
        """極端なパラメータ値でのテスト"""
        print("\n=== 極端なパラメータ値テスト ===")

        extreme_test_cases = [
            # 極端に小さい期間
            {"type": "RSI", "period": 1},
            {"type": "SMA", "period": 2},
            {"type": "EMA", "period": 1},

            # 極端に大きい期間
            {"type": "RSI", "period": 1000},
            {"type": "SMA", "period": 999},
            {"type": "BB", "period": 500},

            # 境界値
            {"type": "RSI", "period": 0},  # 無効値
            {"type": "SMA", "period": -1},  # 負の値
            {"type": "CCI", "period": 10000},  # 非常に大きい値

            # 特殊なパラメータ
            {"type": "BB", "period": 20, "deviation": 0.1},  # 極小偏差
            {"type": "BB", "period": 20, "deviation": 10.0},  # 極大偏差
        ]

        for i, params in enumerate(extreme_test_cases):
            try:
                print(f"\n--- テストケース {i+1}: {params} ---")

                # 極端なパラメータで指標を作成
                indicator = IndicatorGene(
                    type=params["type"],
                    parameters=params,
                    enabled=True
                )

                # 条件生成を実行
                start_time = time.time()
                long_conds, short_conds, exit_conds = self.generator.generate_balanced_conditions([indicator])
                end_time = time.time()

                processing_time = end_time - start_time
                self.test_results["processing_times"].append(processing_time)

                # 基本的な検証
                assert len(long_conds) > 0, "ロング条件が生成されませんでした"
                assert len(short_conds) > 0, "ショート条件が生成されませんでした"
                assert isinstance(long_conds[0], Condition), "ロング条件の型が正しくありません"
                assert isinstance(short_conds[0], Condition), "ショート条件の型が正しくありません"

                print(f"✅ 成功: 処理時間 {processing_time:.4f}秒")
                print(f"   ロング条件数: {len(long_conds)}, ショート条件数: {len(short_conds)}")

                self.test_results["extreme_parameters"]["passed"] += 1

            except Exception as e:
                error_msg = f"パラメータ {params}: {str(e)}"
                print(f"❌ エラー: {error_msg}")
                self.test_results["extreme_parameters"]["failed"] += 1
                self.test_results["extreme_parameters"]["errors"].append(error_msg)

    def test_missing_indicator_data(self):
        """指標データ不足時のテスト"""
        print("\n=== 指標データ不足テスト ===")

        missing_data_cases = [
            # 空のリスト
            [],

            # 無効化された指標のみ
            [IndicatorGene(type="RSI", parameters={"period": 14}, enabled=False)],

            # 存在しない指標タイプ
            [IndicatorGene(type="UNKNOWN_INDICATOR", parameters={"period": 14}, enabled=True)],

            # パラメータが不完全
            [IndicatorGene(type="RSI", parameters={}, enabled=True)],

            # Noneパラメータ
            [IndicatorGene(type="SMA", parameters=None, enabled=True)],

            # 混在ケース（有効・無効・不正）
            [
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                IndicatorGene(type="INVALID", parameters={"period": 20}, enabled=True),
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=False)
            ]
        ]

        for i, indicators in enumerate(missing_data_cases):
            try:
                print(f"\n--- データ不足ケース {i+1} ---")
                print(f"指標数: {len(indicators)}")

                if indicators:
                    enabled_count = sum(1 for ind in indicators if ind.enabled)
                    valid_types = sum(1 for ind in indicators if ind.type in INDICATOR_CHARACTERISTICS)
                    print(f"有効指標数: {enabled_count}, 有効タイプ数: {valid_types}")

                start_time = time.time()
                long_conds, short_conds, exit_conds = self.generator.generate_balanced_conditions(indicators)
                end_time = time.time()

                processing_time = end_time - start_time

                # フォールバック条件が生成されることを確認
                assert len(long_conds) > 0, "フォールバック条件が生成されませんでした"
                assert len(short_conds) > 0, "フォールバック条件が生成されませんでした"

                print(f"✅ 成功: フォールバック条件生成 (処理時間: {processing_time:.4f}秒)")
                print(f"   ロング条件: {[str(c.left_operand) + c.operator + str(c.right_operand) for c in long_conds]}")
                print(f"   ショート条件: {[str(c.left_operand) + c.operator + str(c.right_operand) for c in short_conds]}")

                self.test_results["missing_data"]["passed"] += 1

            except Exception as e:
                error_msg = f"データ不足ケース {i+1}: {str(e)}"
                print(f"❌ エラー: {error_msg}")
                self.test_results["missing_data"]["failed"] += 1
                self.test_results["missing_data"]["errors"].append(error_msg)

    def test_invalid_combinations(self):
        """無効な指標組み合わせでのテスト"""
        print("\n=== 無効な組み合わせテスト ===")

        invalid_combinations = [
            # 同じ指標の大量重複
            [IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)] * 10,

            # 相互に矛盾する指標
            [
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),  # 完全重複
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)
            ],

            # 異常に多い指標数
            [
                IndicatorGene(type=indicator_type, parameters={"period": 14}, enabled=True)
                for indicator_type in ["RSI", "SMA", "EMA", "BB", "CCI", "MACD", "STOCH", "ADX", "ATR"] * 5
            ],

            # 型が混在
            [
                IndicatorGene(type="RSI", parameters={"period": "invalid"}, enabled=True),  # 文字列期間
                IndicatorGene(type="SMA", parameters={"period": 14.5}, enabled=True),  # 小数期間
            ]
        ]

        for i, indicators in enumerate(invalid_combinations):
            try:
                print(f"\n--- 無効組み合わせ {i+1}: {len(indicators)}個の指標 ---")

                start_time = time.time()
                long_conds, short_conds, exit_conds = self.generator.generate_balanced_conditions(indicators)
                end_time = time.time()

                processing_time = end_time - start_time

                # 何らかの条件が生成されることを確認
                assert len(long_conds) > 0, "条件が生成されませんでした"
                assert len(short_conds) > 0, "条件が生成されませんでした"

                print(f"✅ 成功: 処理時間 {processing_time:.4f}秒")
                print(f"   生成された条件数 - ロング: {len(long_conds)}, ショート: {len(short_conds)}")

                # 処理時間が合理的な範囲内であることを確認
                if processing_time > 5.0:  # 5秒以上は異常
                    print(f"⚠️  警告: 処理時間が長すぎます ({processing_time:.2f}秒)")

                self.test_results["invalid_combinations"]["passed"] += 1

            except Exception as e:
                error_msg = f"無効組み合わせ {i+1}: {str(e)}"
                print(f"❌ エラー: {error_msg}")
                self.test_results["invalid_combinations"]["failed"] += 1
                self.test_results["invalid_combinations"]["errors"].append(error_msg)

    def test_performance_limits(self):
        """パフォーマンス限界テスト"""
        print("\n=== パフォーマンス限界テスト ===")

        # メモリ使用量監視開始
        tracemalloc.start()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        performance_test_cases = [
            # 大量の戦略生成
            {
                "name": "大量戦略生成 (100回)",
                "iterations": 100,
                "indicators": [
                    IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                    IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
                ]
            },

            # 複雑な指標組み合わせ
            {
                "name": "複雑な組み合わせ (50回)",
                "iterations": 50,
                "indicators": [
                    IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                    IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
                    IndicatorGene(type="EMA", parameters={"period": 12}, enabled=True),
                    IndicatorGene(type="BB", parameters={"period": 20}, enabled=True),
                    IndicatorGene(type="CCI", parameters={"period": 14}, enabled=True),
                    IndicatorGene(type="ADX", parameters={"period": 14}, enabled=True)
                ]
            },

            # 高頻度実行
            {
                "name": "高頻度実行 (1000回)",
                "iterations": 1000,
                "indicators": [
                    IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)
                ]
            }
        ]

        for test_case in performance_test_cases:
            try:
                print(f"\n--- {test_case['name']} ---")

                times = []
                memory_usage = []

                for i in range(test_case["iterations"]):
                    # メモリ使用量測定
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_usage.append(current_memory)

                    # 処理時間測定
                    start_time = time.time()
                    long_conds, short_conds, exit_conds = self.generator.generate_balanced_conditions(
                        test_case["indicators"]
                    )
                    end_time = time.time()

                    processing_time = end_time - start_time
                    times.append(processing_time)

                    # 進捗表示（10%刻み）
                    if (i + 1) % max(1, test_case["iterations"] // 10) == 0:
                        progress = ((i + 1) / test_case["iterations"]) * 100
                        print(f"   進捗: {progress:.0f}% ({i+1}/{test_case['iterations']})")

                # 統計計算
                avg_time = sum(times) / len(times)
                max_time = max(times)
                min_time = min(times)
                avg_memory = sum(memory_usage) / len(memory_usage)
                peak_memory = max(memory_usage)

                print(f"✅ 完了:")
                print(f"   平均処理時間: {avg_time:.4f}秒")
                print(f"   最大処理時間: {max_time:.4f}秒")
                print(f"   最小処理時間: {min_time:.4f}秒")
                print(f"   平均メモリ使用量: {avg_memory:.2f}MB")
                print(f"   ピークメモリ使用量: {peak_memory:.2f}MB")

                # パフォーマンス基準チェック
                if avg_time > 0.1:  # 平均100ms以上は警告
                    print(f"⚠️  警告: 平均処理時間が長すぎます ({avg_time:.4f}秒)")

                if peak_memory - initial_memory > 100:  # 100MB以上の増加は警告
                    print(f"⚠️  警告: メモリ使用量が大幅に増加しました (+{peak_memory - initial_memory:.2f}MB)")

                # 結果を保存
                self.test_results["performance_limits"]["passed"] += 1
                self.test_results["memory_usage"]["peak_mb"] = max(
                    self.test_results["memory_usage"]["peak_mb"], peak_memory
                )
                self.test_results["memory_usage"]["average_mb"] = avg_memory
                self.test_results["processing_times"].extend(times)

            except Exception as e:
                error_msg = f"{test_case['name']}: {str(e)}"
                print(f"❌ エラー: {error_msg}")
                self.test_results["performance_limits"]["failed"] += 1
                self.test_results["performance_limits"]["errors"].append(error_msg)

        # メモリ使用量監視終了
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"\n=== メモリ使用量サマリー ===")
        print(f"現在のメモリ使用量: {current / 1024 / 1024:.2f}MB")
        print(f"ピークメモリ使用量: {peak / 1024 / 1024:.2f}MB")

    def print_test_summary(self):
        """テスト結果のサマリーを出力"""
        print("\n" + "="*60)
        print("🧪 エッジケーステスト結果サマリー")
        print("="*60)

        total_passed = 0
        total_failed = 0

        for category, results in self.test_results.items():
            if isinstance(results, dict) and "passed" in results:
                passed = results["passed"]
                failed = results["failed"]
                total_passed += passed
                total_failed += failed

                success_rate = (passed / (passed + failed)) * 100 if (passed + failed) > 0 else 0

                print(f"\n📊 {category.replace('_', ' ').title()}:")
                print(f"   成功: {passed}, 失敗: {failed}")
                print(f"   成功率: {success_rate:.1f}%")

                if results["errors"]:
                    print(f"   エラー詳細:")
                    for error in results["errors"][:3]:  # 最初の3つのエラーのみ表示
                        print(f"     - {error}")
                    if len(results["errors"]) > 3:
                        print(f"     ... 他 {len(results['errors']) - 3} 件")

        # パフォーマンス統計
        if self.test_results["processing_times"]:
            times = self.test_results["processing_times"]
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)

            print(f"\n⏱️  処理時間統計:")
            print(f"   平均: {avg_time:.4f}秒")
            print(f"   最大: {max_time:.4f}秒")
            print(f"   最小: {min_time:.4f}秒")
            print(f"   サンプル数: {len(times)}")

        # メモリ使用量
        memory = self.test_results["memory_usage"]
        if memory["peak_mb"] > 0:
            print(f"\n💾 メモリ使用量:")
            print(f"   ピーク: {memory['peak_mb']:.2f}MB")
            print(f"   平均: {memory['average_mb']:.2f}MB")

        # 総合評価
        overall_success_rate = (total_passed / (total_passed + total_failed)) * 100 if (total_passed + total_failed) > 0 else 0

        print(f"\n🎯 総合結果:")
        print(f"   総テスト数: {total_passed + total_failed}")
        print(f"   成功: {total_passed}")
        print(f"   失敗: {total_failed}")
        print(f"   総合成功率: {overall_success_rate:.1f}%")

        # 判定基準
        if overall_success_rate >= 95:
            print(f"\n✅ 判定: 優秀 - 本格運用可能")
        elif overall_success_rate >= 85:
            print(f"\n🟡 判定: 良好 - 軽微な改善後に運用可能")
        elif overall_success_rate >= 70:
            print(f"\n🟠 判定: 要改善 - 問題修正が必要")
        else:
            print(f"\n🔴 判定: 不合格 - 大幅な修正が必要")

        return overall_success_rate


def run_edge_case_tests():
    """エッジケーステストを実行"""
    print("🚀 SmartConditionGenerator エッジケーステスト開始")
    print("="*60)

    test_instance = TestEdgeCases()
    test_instance.setup_method()

    try:
        # 各テストを実行
        test_instance.test_extreme_parameter_values()
        test_instance.test_missing_indicator_data()
        test_instance.test_invalid_combinations()
        test_instance.test_performance_limits()

        # 結果サマリーを表示
        success_rate = test_instance.print_test_summary()

        return success_rate >= 85  # 85%以上で合格

    except Exception as e:
        print(f"\n🚨 テスト実行中に予期しないエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_edge_case_tests()

    if success:
        print("\n🎉 エッジケーステストが成功しました！")
        exit(0)
    else:
        print("\n💥 エッジケーステストで問題が発見されました。")
        exit(1)