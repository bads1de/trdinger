#!/usr/bin/env python3
"""
テスト実行スクリプト

カテゴリ別にテストを実行するためのスクリプトです。
"""

import sys
import os
import subprocess
import argparse
from typing import List, Optional

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_tests(
    category: Optional[str] = None,
    verbose: bool = True,
    coverage: bool = False,
    parallel: bool = False,
    specific_test: Optional[str] = None,
) -> int:
    """
    テストを実行する

    Args:
        category: テストカテゴリ (unit, integration, e2e, slow, etc.)
        verbose: 詳細出力
        coverage: カバレッジレポート生成
        parallel: 並列実行
        specific_test: 特定のテストファイル/メソッド

    Returns:
        終了コード
    """

    # 基本コマンド
    cmd = ["python", "-m", "pytest"]

    # カテゴリ指定
    if category:
        cmd.extend(["-m", category])

    # 特定のテスト指定
    if specific_test:
        cmd.append(specific_test)

    # 詳細出力
    if verbose:
        cmd.append("-v")

    # カバレッジ
    if coverage:
        cmd.extend(["--cov=app", "--cov-report=html", "--cov-report=term"])

    # 並列実行
    if parallel:
        cmd.extend(["-n", "auto"])

    # テストディレクトリ指定
    if not specific_test:
        cmd.append("tests/")

    print(f"実行コマンド: {' '.join(cmd)}")
    print("=" * 60)

    # テスト実行
    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(__file__) + "/..")
        return result.returncode
    except KeyboardInterrupt:
        print("\nテスト実行が中断されました")
        return 1
    except Exception as e:
        print(f"テスト実行中にエラーが発生しました: {e}")
        return 1


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="テスト実行スクリプト")

    parser.add_argument(
        "-c",
        "--category",
        choices=[
            "unit",
            "integration",
            "e2e",
            "slow",
            "market_validation",
            "performance",
            "security",
        ],
        help="テストカテゴリを指定",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", default=True, help="詳細出力"
    )

    parser.add_argument(
        "--coverage", action="store_true", help="カバレッジレポートを生成"
    )

    parser.add_argument("-p", "--parallel", action="store_true", help="並列実行")

    parser.add_argument("-t", "--test", help="特定のテストファイル/メソッドを実行")

    parser.add_argument(
        "--quick", action="store_true", help="高速テスト（unit + integration のみ）"
    )

    parser.add_argument("--full", action="store_true", help="全テスト実行")

    args = parser.parse_args()

    # クイックテスト
    if args.quick:
        print("🚀 高速テスト実行 (unit + integration)")
        exit_code = run_tests(
            "unit or integration", args.verbose, args.coverage, args.parallel
        )
        return exit_code

    # 全テスト
    if args.full:
        print("🚀 全テスト実行")
        exit_code = run_tests(None, args.verbose, args.coverage, args.parallel)
        return exit_code

    # カテゴリ別実行
    if args.category:
        print(f"🚀 {args.category} テスト実行")
        exit_code = run_tests(
            args.category, args.verbose, args.coverage, args.parallel, args.test
        )
        return exit_code

    # 特定テスト実行
    if args.test:
        print(f"🚀 特定テスト実行: {args.test}")
        exit_code = run_tests(
            None, args.verbose, args.coverage, args.parallel, args.test
        )
        return exit_code

    # デフォルト: 高速テスト
    print("🚀 デフォルト実行 (unit + integration)")
    exit_code = run_tests(
        "unit or integration", args.verbose, args.coverage, args.parallel
    )
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
