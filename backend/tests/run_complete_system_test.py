#!/usr/bin/env python3
"""
オートストラテジー強化システム 完全統合テスト

すべての新機能の動作確認と統合テストを実行します。
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header(title: str):
    """ヘッダーを表示"""
    print("\n" + "=" * 80)
    print(f"🚀 {title}")
    print("=" * 80)


def print_section(title: str):
    """セクションヘッダーを表示"""
    print(f"\n📋 {title}")
    print("-" * 60)


def run_test_script(script_path: str, description: str) -> bool:
    """テストスクリプトを実行"""
    try:
        print(f"実行中: {description}")

        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=300,
        )

        if result.returncode == 0:
            print("✅ 成功")
            if result.stdout:
                # 重要な出力のみ表示
                lines = result.stdout.split("\n")
                important_lines = [
                    line
                    for line in lines
                    if any(marker in line for marker in ["✅", "❌", "⚠️", "🎉"])
                ]
                for line in important_lines[-5:]:  # 最後の5行のみ
                    print(f"   {line}")
            return True
        else:
            print("❌ 失敗")
            if result.stderr:
                print(f"   エラー: {result.stderr[:200]}...")
            return False

    except subprocess.TimeoutExpired:
        print("❌ タイムアウト")
        return False
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        return False


def check_system_requirements() -> bool:
    """システム要件をチェック"""
    print_section("システム要件チェック")

    # Python バージョンチェック
    python_version = sys.version_info
    if python_version.major < 3 or (
        python_version.major == 3 and python_version.minor < 8
    ):
        print("❌ Python 3.8以上が必要です")
        return False
    print(
        f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}"
    )

    # 必要ファイルの存在チェック
    required_files = [
        "requirements.txt",
        "app/core/services/feature_engineering/feature_engineering_service.py",
        "app/core/services/ml/signal_generator.py",
        "app/core/services/auto_strategy/services/ml_indicator_service.py",
        "app/core/services/auto_strategy/engines/fitness_sharing.py",
        "app/core/services/monitoring/performance_monitor.py",
        "app/core/services/auto_retraining/auto_retraining_scheduler.py",
        "app/core/services/optimization/bayesian_optimizer.py",
    ]

    missing_files = []
    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("❌ 以下のファイルが見つかりません:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False

    print("✅ 必要ファイルが存在します")
    return True


def run_phase1_tests() -> bool:
    """Phase 1: 基本動作確認とライブラリ統合テスト"""
    print_section("Phase 1: 基本動作確認とライブラリ統合テスト")

    tests = [
        ("tests/test_new_libraries.py", "新ライブラリの動作確認"),
        ("tests/test_ml_indicators.py", "ML指標の動作テスト"),
        ("tests/test_fitness_sharing.py", "フィットネス共有機能の検証"),
        (
            "tests/test_feature_engineering_integration.py",
            "FeatureEngineeringServiceの統合テスト",
        ),
    ]

    success_count = 0
    for script_path, description in tests:
        if run_test_script(script_path, description):
            success_count += 1
        time.sleep(1)  # テスト間の間隔

    success_rate = success_count / len(tests)
    print(f"\nPhase 1 結果: {success_count}/{len(tests)} 成功 ({success_rate:.1%})")

    return success_rate >= 0.75  # 75%以上の成功率


def run_phase2_tests() -> bool:
    """Phase 2: バックテスト性能評価"""
    print_section("Phase 2: バックテスト性能評価")

    tests = [("tests/test_backtest_performance_comparison.py", "バックテスト性能比較")]

    success_count = 0
    for script_path, description in tests:
        if run_test_script(script_path, description):
            success_count += 1
        time.sleep(1)

    success_rate = success_count / len(tests)
    print(f"\nPhase 2 結果: {success_count}/{len(tests)} 成功 ({success_rate:.1%})")

    return success_rate >= 0.75


def test_phase4_components() -> bool:
    """Phase 4: 継続的改善システムのコンポーネントテスト"""
    print_section("Phase 4: 継続的改善システムのコンポーネントテスト")

    try:
        # パフォーマンス監視システムのテスト
        print("パフォーマンス監視システムのテスト...")
        from app.core.services.monitoring import PerformanceMonitor

        monitor = PerformanceMonitor()
        test_data = {
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.08,
            "win_rate": 0.6,
            "total_trades": 50,
            "long_trades": 30,
            "short_trades": 20,
            "long_pnl": 8000,
            "short_pnl": 7000,
            "balance_score": 0.8,
        }

        monitor.add_performance_record("test_strategy", test_data)
        performance = monitor.get_strategy_performance("test_strategy")

        if performance:
            print("✅ パフォーマンス監視システム")
        else:
            print("❌ パフォーマンス監視システム")
            return False

        # 自動再学習スケジューラーのテスト
        print("自動再学習スケジューラーのテスト...")
        from app.core.services.auto_retraining import AutoRetrainingScheduler

        scheduler = AutoRetrainingScheduler()
        job_id = scheduler.trigger_immediate_retraining("test_model", "test")
        job_status = scheduler.get_job_status(job_id)

        if job_status:
            print("✅ 自動再学習スケジューラー")
        else:
            print("❌ 自動再学習スケジューラー")
            return False

        # ベイズ最適化エンジンのテスト
        print("ベイズ最適化エンジンのテスト...")
        from app.core.services.optimization import BayesianOptimizer

        optimizer = BayesianOptimizer()

        # 簡単な目的関数でテスト
        def test_objective(params):
            return -((params.get("x", 0) - 0.5) ** 2)  # x=0.5で最大

        # フォールバック最適化でテスト（scikit-optimizeがない場合）
        result = optimizer._optimize_with_fallback(
            test_objective, {"x": {"type": "real", "low": 0, "high": 1}}, 10
        )

        if result and "best_params" in result:
            print("✅ ベイズ最適化エンジン")
        else:
            print("❌ ベイズ最適化エンジン")
            return False

        print("\nPhase 4 結果: 3/3 成功 (100%)")
        return True

    except Exception as e:
        print(f"❌ Phase 4 テストエラー: {e}")
        return False


def generate_summary_report(
    phase1_success: bool, phase2_success: bool, phase4_success: bool
):
    """サマリーレポートを生成"""
    print_header("オートストラテジー強化システム 統合テスト結果")

    total_phases = 3
    successful_phases = sum([phase1_success, phase2_success, phase4_success])

    print(f"📊 総合結果: {successful_phases}/{total_phases} フェーズ成功")
    print()

    print("📋 フェーズ別結果:")
    print(f"   Phase 1 (基本動作確認): {'✅ 成功' if phase1_success else '❌ 失敗'}")
    print(
        f"   Phase 2 (ショート戦略強化): {'✅ 成功' if phase2_success else '❌ 失敗'}"
    )
    print(f"   Phase 4 (継続的改善): {'✅ 成功' if phase4_success else '❌ 失敗'}")
    print()

    if successful_phases == total_phases:
        print("🎉 すべてのフェーズが成功しました！")
        print("オートストラテジー強化システムは本格運用の準備が整っています。")
        print()
        print("📚 次のステップ:")
        print(
            "   1. 運用ガイド（docs/ENHANCED_AUTO_STRATEGY_OPERATION_GUIDE.md）を確認"
        )
        print("   2. 本番環境での段階的導入")
        print("   3. パフォーマンス監視の開始")
        print("   4. 定期的なMLモデル再学習の設定")
    else:
        print("⚠️ 一部のフェーズで問題が発生しました。")
        print("失敗したフェーズのログを確認し、問題を解決してください。")

    return successful_phases == total_phases


def main():
    """メイン実行関数"""
    print_header("オートストラテジー強化システム 完全統合テスト")

    start_time = time.time()

    # システム要件チェック
    if not check_system_requirements():
        print("❌ システム要件を満たしていません")
        return False

    # Phase 1 テスト
    phase1_success = run_phase1_tests()

    # Phase 2 テスト
    phase2_success = run_phase2_tests()

    # Phase 4 テスト
    phase4_success = test_phase4_components()

    # サマリーレポート生成
    overall_success = generate_summary_report(
        phase1_success, phase2_success, phase4_success
    )

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n⏱️ 総実行時間: {total_time:.1f}秒")

    return overall_success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによって中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 予期しないエラーが発生しました: {e}")
        sys.exit(1)
