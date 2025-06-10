#!/usr/bin/env python3
"""
テクニカル指標サービス マスター包括テスト

全てのテストを統合して実行し、包括的な結果レポートを生成します。
"""

import sys
import os
import subprocess
import time
from datetime import datetime

# バックエンドのパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_test_file(test_file_path, test_name):
    """個別のテストファイルを実行"""
    print(f"\n🧪 {test_name} 実行中...")
    print("=" * 60)

    try:
        start_time = time.time()

        # Pythonスクリプトとして実行
        result = subprocess.run(
            [sys.executable, test_file_path],
            capture_output=True,
            text=True,
            timeout=300,  # 5分のタイムアウト
        )

        end_time = time.time()
        execution_time = end_time - start_time

        print(result.stdout)

        if result.stderr:
            print("⚠️ エラー出力:")
            print(result.stderr)

        success = result.returncode == 0

        print(f"\n⏱️ 実行時間: {execution_time:.2f}秒")
        print(f"📊 結果: {'✅ 成功' if success else '❌ 失敗'}")

        return success, execution_time, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        print("❌ テストがタイムアウトしました")
        return False, 300, "", "Timeout"
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        return False, 0, "", str(e)


def generate_report(test_results):
    """テスト結果レポートを生成"""
    report_lines = []
    report_lines.append("# テクニカル指標サービス 包括テスト結果レポート")
    report_lines.append(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # サマリー
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result["success"])
    total_time = sum(result["execution_time"] for result in test_results.values())

    report_lines.append("## 📊 テスト結果サマリー")
    report_lines.append(f"- 総テスト数: {total_tests}")
    report_lines.append(f"- 成功: {passed_tests}")
    report_lines.append(f"- 失敗: {total_tests - passed_tests}")
    report_lines.append(f"- 成功率: {(passed_tests/total_tests)*100:.1f}%")
    report_lines.append(f"- 総実行時間: {total_time:.2f}秒")
    report_lines.append("")

    # 個別テスト結果
    report_lines.append("## 📋 個別テスト結果")
    for test_name, result in test_results.items():
        status = "✅ 成功" if result["success"] else "❌ 失敗"
        report_lines.append(f"### {test_name}")
        report_lines.append(f"- 結果: {status}")
        report_lines.append(f"- 実行時間: {result['execution_time']:.2f}秒")

        if not result["success"]:
            report_lines.append("- エラー詳細:")
            report_lines.append("```")
            report_lines.append(result["stderr"])
            report_lines.append("```")

        report_lines.append("")

    # 推奨事項
    report_lines.append("## 🔧 推奨事項")
    if passed_tests == total_tests:
        report_lines.append(
            "✅ 全てのテストが成功しました。システムは正常に動作しています。"
        )
        report_lines.append("")
        report_lines.append("次のステップ:")
        report_lines.append("1. 本番環境でのデプロイメント準備")
        report_lines.append("2. パフォーマンス監視の設定")
        report_lines.append("3. 定期的なテスト実行の自動化")
    else:
        report_lines.append("⚠️ 一部のテストが失敗しています。以下の対応が必要です:")
        report_lines.append("")
        for test_name, result in test_results.items():
            if not result["success"]:
                report_lines.append(f"- {test_name}: 修正が必要")
        report_lines.append("")
        report_lines.append("修正後に再度テストを実行してください。")

    return "\n".join(report_lines)


def main():
    """メインテスト実行関数"""
    print("🔬 テクニカル指標サービス マスター包括テスト")
    print("=" * 80)
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # テストファイルの定義
    test_files = {
        "TA-Lib基本動作確認": "backend/tests/check_talib.py",
        "TA-Lib移行包括的テスト": "backend/tests/comprehensive_test.py",
        "IndicatorOrchestrator": "backend/tests/test_indicator_orchestrator.py",
        "個別指標クラス": "backend/tests/test_individual_indicators.py",
        "統合・エラーハンドリング": "backend/tests/test_integration_and_errors.py",
    }

    test_results = {}

    # 各テストファイルを実行
    for test_name, test_file in test_files.items():
        if os.path.exists(test_file):
            success, execution_time, stdout, stderr = run_test_file(
                test_file, test_name
            )
            test_results[test_name] = {
                "success": success,
                "execution_time": execution_time,
                "stdout": stdout,
                "stderr": stderr,
            }
        else:
            print(f"⚠️ テストファイルが見つかりません: {test_file}")
            test_results[test_name] = {
                "success": False,
                "execution_time": 0,
                "stdout": "",
                "stderr": f"File not found: {test_file}",
            }

    # 結果サマリー表示
    print("\n" + "=" * 80)
    print("📋 最終結果サマリー")
    print("=" * 80)

    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result["success"])
    total_time = sum(result["execution_time"] for result in test_results.values())

    for test_name, result in test_results.items():
        status = "✅ 成功" if result["success"] else "❌ 失敗"
        print(f"{test_name}: {status} ({result['execution_time']:.2f}秒)")

    print(f"\n📊 総合結果: {passed_tests}/{total_tests} 成功")
    print(f"⏱️ 総実行時間: {total_time:.2f}秒")
    print(f"📈 成功率: {(passed_tests/total_tests)*100:.1f}%")

    # レポート生成
    report_content = generate_report(test_results)

    # レポートファイルに保存
    report_file = "backend/tests/COMPREHENSIVE_TEST_REPORT.md"
    try:
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"\n📄 詳細レポートを保存しました: {report_file}")
    except Exception as e:
        print(f"⚠️ レポート保存エラー: {e}")

    # 最終判定
    if passed_tests == total_tests:
        print("\n🎉 全てのテストが成功しました！")
        print("テクニカル指標サービスは正常に動作しています。")
        return 0
    else:
        print(f"\n⚠️ {total_tests - passed_tests}個のテストが失敗しました。")
        print("修正が必要です。詳細はレポートを確認してください。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
