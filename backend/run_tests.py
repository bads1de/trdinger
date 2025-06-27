#!/usr/bin/env python3
"""
ユニットテスト実行スクリプト
"""

import pytest
import sys
import os


def run_unit_tests():
    """ユニットテストを実行"""
    print("ストラテジービルダー機能のユニットテストを実行します\n")

    # テストファイルのパス
    test_files = [
        "tests/test_strategy_builder_service.py",
        "tests/test_user_strategy_repository.py",
    ]

    # 各テストファイルが存在するかチェック
    missing_files = []
    for test_file in test_files:
        if not os.path.exists(test_file):
            missing_files.append(test_file)

    if missing_files:
        print("以下のテストファイルが見つかりません:")
        for file in missing_files:
            print(f"  - {file}")
        return False

    # pytestを実行
    try:
        # 詳細な出力でテストを実行
        exit_code = pytest.main(
            [
                "-v",  # 詳細出力
                "--tb=short",  # トレースバックを短縮
                "--color=yes",  # カラー出力
                *test_files,
            ]
        )

        if exit_code == 0:
            print("\nすべてのユニットテストが成功しました！")
            return True
        else:
            print(f"\n一部のテストが失敗しました (終了コード: {exit_code})")
            return False

    except Exception as e:
        print(f"\nテスト実行中にエラーが発生しました: {e}")
        return False


def main():
    """メイン関数"""
    success = run_unit_tests()

    if success:
        print("\nユニットテストが正常に完了しました。")
        sys.exit(0)
    else:
        print("\nユニットテストで問題が発生しました。")
        sys.exit(1)


if __name__ == "__main__":
    main()
