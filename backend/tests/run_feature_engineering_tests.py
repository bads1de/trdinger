#!/usr/bin/env python3
"""
特徴量エンジニアリング包括テスト実行スクリプト

このスクリプトは特徴量エンジニアリング機能の包括的なテストを実行し、
結果をレポート形式で出力します。

使用方法:
    python backend/tests/ml/run_feature_engineering_tests.py [オプション]

オプション:
    --verbose, -v     詳細出力
    --quiet, -q       簡潔出力
    --coverage        カバレッジレポート生成
    --html            HTMLレポート生成
    --benchmark       ベンチマークテスト実行
"""

import sys
import os
import argparse
import subprocess
import time
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_command(command, capture_output=True):
    """コマンドを実行し、結果を返す"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=capture_output,
            text=True,
            cwd=project_root
        )
        return result
    except Exception as e:
        print(f"❌ コマンド実行エラー: {e}")
        return None

def print_header(title):
    """ヘッダーを出力"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_section(title):
    """セクションヘッダーを出力"""
    print(f"\n📋 {title}")
    print("-" * 40)

def run_basic_tests(verbose=False):
    """基本テストを実行"""
    print_section("基本テスト実行")
    
    test_file = "backend/tests/ml/test_feature_engineering_comprehensive.py"
    
    if verbose:
        command = f"python -m pytest {test_file} -v --tb=short"
    else:
        command = f"python -m pytest {test_file} --tb=short"
    
    print(f"実行コマンド: {command}")
    
    start_time = time.time()
    result = run_command(command, capture_output=False)
    end_time = time.time()
    
    if result and result.returncode == 0:
        print(f"\n✅ 全テスト成功 (実行時間: {end_time - start_time:.2f}秒)")
        return True
    else:
        print(f"\n❌ テスト失敗 (実行時間: {end_time - start_time:.2f}秒)")
        return False

def run_coverage_tests():
    """カバレッジテストを実行"""
    print_section("カバレッジテスト実行")
    
    # カバレッジ測定
    command = (
        "python -m pytest "
        "backend/tests/ml/test_feature_engineering_comprehensive.py "
        "--cov=backend.app.core.services.ml.feature_engineering "
        "--cov-report=term-missing "
        "--cov-report=html:backend/tests/ml/htmlcov"
    )
    
    print(f"実行コマンド: {command}")
    result = run_command(command, capture_output=False)
    
    if result and result.returncode == 0:
        print("\n✅ カバレッジレポート生成完了")
        print("📄 HTMLレポート: backend/tests/ml/htmlcov/index.html")
        return True
    else:
        print("\n❌ カバレッジテスト失敗")
        return False

def run_benchmark_tests():
    """ベンチマークテストを実行"""
    print_section("ベンチマークテスト実行")
    
    command = (
        "python -m pytest "
        "backend/tests/ml/test_feature_engineering_comprehensive.py "
        "--durations=10 "
        "--benchmark-only"
    )
    
    print(f"実行コマンド: {command}")
    result = run_command(command, capture_output=False)
    
    if result and result.returncode == 0:
        print("\n✅ ベンチマークテスト完了")
        return True
    else:
        print("\n❌ ベンチマークテスト失敗")
        return False

def generate_html_report():
    """HTMLレポートを生成"""
    print_section("HTMLレポート生成")
    
    command = (
        "python -m pytest "
        "backend/tests/ml/test_feature_engineering_comprehensive.py "
        "--html=backend/tests/ml/report.html "
        "--self-contained-html"
    )
    
    print(f"実行コマンド: {command}")
    result = run_command(command, capture_output=False)
    
    if result and result.returncode == 0:
        print("\n✅ HTMLレポート生成完了")
        print("📄 レポート: backend/tests/ml/report.html")
        return True
    else:
        print("\n❌ HTMLレポート生成失敗")
        return False

def check_dependencies():
    """必要な依存関係をチェック"""
    print_section("依存関係チェック")
    
    required_packages = [
        "pytest",
        "pandas",
        "numpy",
        "scikit-learn"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (未インストール)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  不足パッケージ: {', '.join(missing_packages)}")
        print("pip install で必要なパッケージをインストールしてください。")
        return False
    
    print("\n✅ 全ての依存関係が満たされています")
    return True

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="特徴量エンジニアリング包括テスト実行スクリプト"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="詳細出力"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="簡潔出力"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="カバレッジレポート生成"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="HTMLレポート生成"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="ベンチマークテスト実行"
    )
    
    args = parser.parse_args()
    
    print_header("特徴量エンジニアリング包括テスト")
    print(f"📅 実行日時: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 作業ディレクトリ: {project_root}")
    
    # 依存関係チェック
    if not check_dependencies():
        sys.exit(1)
    
    success = True
    
    # 基本テスト実行
    if not run_basic_tests(verbose=args.verbose):
        success = False
    
    # カバレッジテスト
    if args.coverage and success:
        if not run_coverage_tests():
            success = False
    
    # HTMLレポート生成
    if args.html and success:
        if not generate_html_report():
            success = False
    
    # ベンチマークテスト
    if args.benchmark and success:
        if not run_benchmark_tests():
            success = False
    
    # 結果サマリー
    print_header("実行結果サマリー")
    
    if success:
        print("🎉 全てのテストが正常に完了しました！")
        print("\n📋 生成されたファイル:")
        
        report_files = [
            "backend/tests/ml/feature_engineering_test_report.md",
        ]
        
        if args.coverage:
            report_files.append("backend/tests/ml/htmlcov/index.html")
        
        if args.html:
            report_files.append("backend/tests/ml/report.html")
        
        for file_path in report_files:
            if os.path.exists(file_path):
                print(f"  📄 {file_path}")
        
        print("\n💡 推奨事項:")
        print("  - テストレポートを確認してください")
        print("  - 警告メッセージがある場合は対応を検討してください")
        print("  - 定期的にテストを実行して品質を維持してください")
        
    else:
        print("❌ テスト実行中にエラーが発生しました")
        print("詳細なエラー情報を確認し、問題を修正してください")
        sys.exit(1)

if __name__ == "__main__":
    main()
