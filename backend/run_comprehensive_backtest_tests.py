#!/usr/bin/env python3
"""
包括的バックテストテスト実行スクリプト

実際のDBデータを使用した包括的なバックテストテストを実行します。
"""

import os
import sys
import subprocess
import time
from datetime import datetime
import argparse


def run_command(command, description):
    """コマンドを実行し、結果を表示"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"実行コマンド: {command}")
    print()

    start_time = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    end_time = time.time()

    execution_time = end_time - start_time

    print(f"実行時間: {execution_time:.2f}秒")
    print(f"終了コード: {result.returncode}")

    if result.stdout:
        print("\n📊 標準出力:")
        print(result.stdout)

    if result.stderr:
        print("\n❌ エラー出力:")
        print(result.stderr)

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="包括的バックテストテスト実行")
    parser.add_argument(
        "--quick", action="store_true", help="クイックテスト（基本テストのみ）"
    )
    parser.add_argument(
        "--performance", action="store_true", help="パフォーマンステストのみ"
    )
    parser.add_argument(
        "--error-handling", action="store_true", help="エラーハンドリングテストのみ"
    )
    parser.add_argument("--integration", action="store_true", help="統合テストのみ")
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細出力")

    args = parser.parse_args()

    # ログディレクトリの作成
    log_dir = "tests/logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    print("🧪 包括的バックテストテストスイート")
    print(f"開始時刻: {datetime.now()}")
    print(f"作業ディレクトリ: {os.getcwd()}")

    # 基本的なオプション
    verbose_flag = "-v" if args.verbose else ""

    # テスト結果を記録
    test_results = []

    # 1. データベース接続とデータ存在確認（データ可用性テストで代替）
    if not args.performance and not args.error_handling:
        success = run_command(
            f"python -m pytest tests/integration/test_comprehensive_backtest_with_real_data.py::TestComprehensiveBacktestWithRealData::test_data_availability {verbose_flag}",
            "データベース接続・データ可用性確認",
        )
        test_results.append(("データベース接続・データ可用性", success))

    # 2. 基本的な統合テスト
    if args.integration or not any([args.quick, args.performance, args.error_handling]):
        # データ可用性は上記で実行済みなのでスキップ

        success = run_command(
            f"python -m pytest tests/integration/test_comprehensive_backtest_with_real_data.py::TestComprehensiveBacktestWithRealData::test_data_quality_validation {verbose_flag}",
            "データ品質テスト",
        )
        test_results.append(("データ品質", success))

        success = run_command(
            f"python -m pytest tests/integration/test_comprehensive_backtest_with_real_data.py::TestComprehensiveBacktestWithRealData::test_basic_sma_cross_strategy_btc_usdt {verbose_flag}",
            "基本SMAクロス戦略テスト",
        )
        test_results.append(("基本戦略", success))

    # 3. データ標準化テスト
    if args.integration or not any([args.quick, args.performance, args.error_handling]):
        success = run_command(
            f"python -m pytest tests/integration/test_comprehensive_backtest_with_real_data.py::TestComprehensiveBacktestWithRealData::test_data_standardization_with_real_data {verbose_flag}",
            "データ標準化テスト",
        )
        test_results.append(("データ標準化", success))

    # 4. runner.py統合テスト
    if args.integration or not any([args.quick, args.performance, args.error_handling]):
        success = run_command(
            f"python -m pytest tests/integration/test_comprehensive_backtest_with_real_data.py::TestComprehensiveBacktestWithRealData::test_runner_integration_with_real_data {verbose_flag}",
            "runner.py統合テスト",
        )
        test_results.append(("runner.py統合", success))

    # 5. パフォーマンステスト
    if args.performance or not any([args.quick, args.integration, args.error_handling]):
        success = run_command(
            f"python -m pytest tests/performance/test_backtest_performance_with_real_data.py::TestBacktestPerformanceWithRealData::test_small_dataset_performance {verbose_flag}",
            "小規模データセットパフォーマンス",
        )
        test_results.append(("小規模パフォーマンス", success))

        success = run_command(
            f"python -m pytest tests/performance/test_backtest_performance_with_real_data.py::TestBacktestPerformanceWithRealData::test_medium_dataset_performance {verbose_flag}",
            "中規模データセットパフォーマンス",
        )
        test_results.append(("中規模パフォーマンス", success))

        if not args.quick:
            success = run_command(
                f"python -m pytest tests/performance/test_backtest_performance_with_real_data.py::TestBacktestPerformanceWithRealData::test_large_dataset_performance {verbose_flag}",
                "大規模データセットパフォーマンス",
            )
            test_results.append(("大規模パフォーマンス", success))

    # 6. エラーハンドリングテスト
    if args.error_handling or not any([args.quick, args.performance, args.integration]):
        success = run_command(
            f"python -m pytest tests/integration/test_backtest_error_handling_with_real_data.py::TestBacktestErrorHandlingWithRealData::test_invalid_date_range_handling {verbose_flag}",
            "無効日付範囲エラーハンドリング",
        )
        test_results.append(("無効日付範囲", success))

        success = run_command(
            f"python -m pytest tests/integration/test_backtest_error_handling_with_real_data.py::TestBacktestErrorHandlingWithRealData::test_invalid_strategy_parameters {verbose_flag}",
            "無効戦略パラメータエラーハンドリング",
        )
        test_results.append(("無効戦略パラメータ", success))

        success = run_command(
            f"python -m pytest tests/integration/test_backtest_error_handling_with_real_data.py::TestBacktestErrorHandlingWithRealData::test_invalid_financial_parameters {verbose_flag}",
            "無効金融パラメータエラーハンドリング",
        )
        test_results.append(("無効金融パラメータ", success))

    # 7. 高度なテスト（クイックモードでない場合）
    if not args.quick and (
        args.integration or not any([args.performance, args.error_handling])
    ):
        success = run_command(
            f"python -m pytest tests/integration/test_comprehensive_backtest_with_real_data.py::TestComprehensiveBacktestWithRealData::test_different_symbols_comparison {verbose_flag}",
            "複数シンボル比較テスト",
        )
        test_results.append(("複数シンボル比較", success))

        success = run_command(
            f"python -m pytest tests/integration/test_comprehensive_backtest_with_real_data.py::TestComprehensiveBacktestWithRealData::test_parameter_optimization_with_real_data {verbose_flag}",
            "パラメータ最適化テスト",
        )
        test_results.append(("パラメータ最適化", success))

        success = run_command(
            f"python -m pytest tests/integration/test_comprehensive_backtest_with_real_data.py::TestComprehensiveBacktestWithRealData::test_long_term_strategy_performance {verbose_flag}",
            "長期戦略パフォーマンステスト",
        )
        test_results.append(("長期戦略", success))

    # 8. 全体テスト（特定のテストが指定されていない場合）
    if not any([args.quick, args.performance, args.error_handling, args.integration]):
        print(f"\n{'='*60}")
        print("🎯 全体統合テスト実行")
        print(f"{'='*60}")

        success = run_command(
            f"python -m pytest tests/integration/test_comprehensive_backtest_with_real_data.py {verbose_flag}",
            "全統合テスト",
        )
        test_results.append(("全統合テスト", success))

    # 結果サマリー
    print(f"\n{'='*60}")
    print("📊 テスト結果サマリー")
    print(f"{'='*60}")

    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)

    print(f"総テスト数: {total}")
    print(f"成功: {passed}")
    print(f"失敗: {total - passed}")
    print(f"成功率: {passed/total*100:.1f}%" if total > 0 else "N/A")

    print("\n詳細結果:")
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_name}")

    print(f"\n終了時刻: {datetime.now()}")

    # 失敗があった場合は終了コード1で終了
    if total - passed > 0:
        print("\n⚠️  一部のテストが失敗しました。詳細は上記のログを確認してください。")
        sys.exit(1)
    else:
        print("\n🎉 全てのテストが成功しました！")
        sys.exit(0)


if __name__ == "__main__":
    main()
