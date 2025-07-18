#!/usr/bin/env python3
"""
UnifiedErrorHandler の動作確認テスト

リファクタリング後の統一エラーハンドリングが正常に動作することを確認します。
"""

import sys
import os
import traceback
from typing import Dict, Any

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    from app.core.utils.unified_error_handler import (
        UnifiedErrorHandler,
        UnifiedTimeoutError,
        UnifiedValidationError,
        unified_safe_operation,
        unified_timeout_decorator,
        # 標準エイリアス
        safe_ml_operation,
        timeout_decorator,
    )

    print("✅ UnifiedErrorHandler のインポート成功")
except ImportError as e:
    print(f"❌ インポートエラー: {e}")
    sys.exit(1)


def test_error_response_generation():
    """エラーレスポンス生成のテスト"""
    print("\n🧪 エラーレスポンス生成テスト")

    try:
        response = UnifiedErrorHandler.create_error_response(
            message="テストエラー",
            error_code="TEST_ERROR",
            context="テストコンテキスト",
        )

        assert response["success"] is False
        assert response["message"] == "テストエラー"
        assert response["error_code"] == "TEST_ERROR"
        assert response["context"] == "テストコンテキスト"
        assert "timestamp" in response

        print("✅ エラーレスポンス生成テスト成功")
        return True
    except Exception as e:
        print(f"❌ エラーレスポンス生成テスト失敗: {e}")
        return False


def test_safe_execute():
    """安全実行機能のテスト"""
    print("\n🧪 安全実行機能テスト")

    try:
        # 正常実行のテスト
        def normal_function():
            return "正常実行"

        result = UnifiedErrorHandler.safe_execute(normal_function)
        assert result == "正常実行"

        # エラー時のデフォルト値テスト
        def error_function():
            raise ValueError("テストエラー")

        result = UnifiedErrorHandler.safe_execute(
            error_function, default_return="デフォルト値"
        )
        assert result == "デフォルト値"

        print("✅ 安全実行機能テスト成功")
        return True
    except Exception as e:
        print(f"❌ 安全実行機能テスト失敗: {e}")
        return False


def test_validation_functions():
    """バリデーション機能のテスト"""
    print("\n🧪 バリデーション機能テスト")

    try:
        # 予測値バリデーションテスト
        valid_predictions = {"UP": 0.7, "DOWN": 0.2, "RANGE": 0.1}
        assert UnifiedErrorHandler.validate_predictions(valid_predictions) is True

        invalid_predictions = {"UP": 1.5, "DOWN": -0.1}  # 範囲外
        assert UnifiedErrorHandler.validate_predictions(invalid_predictions) is False

        # データフレームバリデーションテスト（pandas が利用可能な場合）
        try:
            import pandas as pd
            import numpy as np

            valid_df = pd.DataFrame(
                {
                    "open": [1, 2, 3],
                    "high": [2, 3, 4],
                    "low": [0.5, 1.5, 2.5],
                    "close": [1.5, 2.5, 3.5],
                    "volume": [100, 200, 300],
                }
            )

            assert (
                UnifiedErrorHandler.validate_dataframe(
                    valid_df,
                    required_columns=["open", "high", "low", "close", "volume"],
                )
                is True
            )

            empty_df = pd.DataFrame()
            assert UnifiedErrorHandler.validate_dataframe(empty_df) is False

        except ImportError:
            print("⚠️ pandas が利用できないため、データフレームテストをスキップ")

        print("✅ バリデーション機能テスト成功")
        return True
    except Exception as e:
        print(f"❌ バリデーション機能テスト失敗: {e}")
        return False


def test_unified_interface():
    """統一インターフェースのテスト"""
    print("\n🧪 統一インターフェーステスト")

    try:
        # UnifiedErrorHandler の基本機能テスト
        assert UnifiedErrorHandler is not None

        # デコレータエイリアスのテスト
        @safe_ml_operation(default_return="デフォルト")
        def test_function():
            return "成功"

        result = test_function()
        assert result == "成功"

        print("✅ 統一インターフェーステスト成功")
        return True
    except Exception as e:
        print(f"❌ 統一インターフェーステスト失敗: {e}")
        return False


def test_unified_decorators():
    """統一デコレータのテスト"""
    print("\n🧪 統一デコレータテスト")

    try:

        @unified_safe_operation(default_return="エラー時デフォルト")
        def test_decorator_function():
            return "デコレータ成功"

        result = test_decorator_function()
        assert result == "デコレータ成功"

        @unified_safe_operation(default_return="エラー時デフォルト")
        def test_error_function():
            raise RuntimeError("テストエラー")

        result = test_error_function()
        assert result == "エラー時デフォルト"

        print("✅ 統一デコレータテスト成功")
        return True
    except Exception as e:
        print(f"❌ 統一デコレータテスト失敗: {e}")
        return False


def main():
    """メインテスト実行"""
    print("🚀 UnifiedErrorHandler 動作確認テスト開始")
    print("=" * 50)

    tests = [
        test_error_response_generation,
        test_safe_execute,
        test_validation_functions,
        test_unified_interface,
        test_unified_decorators,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ テスト実行エラー: {e}")
            print(traceback.format_exc())
            failed += 1

    print("\n" + "=" * 50)
    print(f"📊 テスト結果: {passed} 成功, {failed} 失敗")

    if failed == 0:
        print("🎉 全てのテストが成功しました！")
        return 0
    else:
        print("⚠️ 一部のテストが失敗しました。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
