"""error_handlingクラスのテストモジュール"""

import pytest
from unittest.mock import patch
from app.services.auto_strategy.utils.error_handling import AutoStrategyErrorHandler, ErrorContext

class TestAutoStrategyErrorHandler:
    """AutoStrategyErrorHandlerのテスト"""

    def test_handle_ga_error(self):
        """GAエラーハンドリングテスト"""
        error = ValueError("Test GA error")
        result = AutoStrategyErrorHandler.handle_ga_error(error, context="Test GA")

        assert result["error_code"] == "GA_ERROR"
        assert "Test GA error" in result["message"]
        assert result["context"] == "Test GA"

    def test_handle_strategy_generation_error(self):
        """戦略生成エラーハンドリングテスト"""
        error = Exception("Strategy error")
        strategy_data = {"type": "test"}
        result = AutoStrategyErrorHandler.handle_strategy_generation_error(
            error, strategy_data=strategy_data, context="Test strategy"
        )

        assert result["success"] is False
        assert result["error"] == "Strategy error"
        assert result["context"] == "Test strategy"
        assert result["strategy_data"] == strategy_data

    def test_handle_calculation_error(self):
        """計算エラーハンドリングテスト（バグ発見目的）"""
        error = ZeroDivisionError("Division by zero")
        result = AutoStrategyErrorHandler.handle_calculation_error(
            error, context="Calculation", fallback_value=42
        )

        # lambda: None が何もしないため、fallback_valueが返るはず
        assert result == 42

    def test_handle_calculation_error_with_log_level(self):
        """ログレベル付き計算エラーハンドリングテスト"""
        error = ValueError("Test calc error")
        result = AutoStrategyErrorHandler.handle_calculation_error(
            error, context="Calc", fallback_value="default", log_level="warning"
        )

        assert result == "default"

    def test_handle_system_error_without_traceback(self):
        """トレースバックなしのシステムエラーハンドリングテスト"""
        error = RuntimeError("System error")
        result = AutoStrategyErrorHandler.handle_system_error(
            error, context="System", include_traceback=False
        )

        assert result["success"] is False
        assert result["error"] == "System error"
        assert result["context"] == "System"
        assert "traceback" not in result

    def test_handle_system_error_with_traceback(self):
        """トレースバック付きのシステムエラーハンドリングテスト"""
        error = Exception("With traceback")
        result = AutoStrategyErrorHandler.handle_system_error(
            error, context="System", include_traceback=True
        )

        assert result["success"] is False
        assert result["error"] == "With traceback"
        assert "traceback" in result

    def test_inheritance_from_error_handler(self):
        """ErrorHandlerからの継承テスト"""
        # safe_executeメソッドが使用可能か確認
        result = AutoStrategyErrorHandler.safe_execute(lambda: "test")
        assert result == "test"


class TestErrorContext:
    """ErrorContextクラスのテスト"""

    def test_initialization(self):
        """初期化テスト"""
        context = ErrorContext("Test context")
        assert context.context == "Test context"
        assert context.errors == []
        assert context.warnings == []

    def test_has_errors_no_errors(self):
        """エラーなしの場合のhas_errorsテスト"""
        context = ErrorContext("Test")
        assert context.has_errors() is False

    def test_has_errors_with_errors(self):
        """エラーありの場合のhas_errorsテスト"""
        context = ErrorContext("Test")
        context.errors = ["error1"]
        assert context.has_errors() is True

    def test_has_warnings_no_warnings(self):
        """警告なしの場合のhas_warningsテスト"""
        context = ErrorContext("Test")
        assert context.has_warnings() is False

    def test_has_warnings_with_warnings(self):
        """警告ありの場合のhas_warningsテスト"""
        context = ErrorContext("Test")
        context.warnings = ["warning1"]
        assert context.has_warnings() is True

    def test_clear(self):
        """clearメソッドのテスト"""
        context = ErrorContext("Test")
        context.errors = ["e1", "e2"]
        context.warnings = ["w1", "w2"]
        context.clear()
        assert context.errors == []
        assert context.warnings == []