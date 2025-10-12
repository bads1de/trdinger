"""
エラー処理統合のテストモジュール

ErrorHandlerと統一エラーハンドリング機能をテストする。
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from fastapi import HTTPException

from backend.app.utils.error_handler import (
    ErrorHandler,
    safe_operation,
    timeout_decorator,
    operation_context,
    safe_ml_operation,
    TimeoutError,
    ValidationError,
    DataError,
    ModelError,
)


class TestErrorHandler:
    """ErrorHandlerクラスのテスト"""

    def test_create_error_response(self):
        """エラーレスポンス生成テスト"""
        response = ErrorHandler.create_error_response(
            message="テストエラー",
            error_code="TEST_ERROR",
            details={"key": "value"},
            context="テストコンテキスト"
        )

        assert isinstance(response, dict)
        assert "message" in response
        assert "error_code" in response
        assert response["message"] == "テストエラー"
        assert response["error_code"] == "TEST_ERROR"

    def test_handle_api_error(self):
        """APIエラーハンドリングテスト"""
        error = ValueError("テストエラー")

        with patch('backend.app.utils.error_handler.logger') as mock_logger:
            http_exception = ErrorHandler.handle_api_error(
                error,
                context="テストコンテキスト",
                status_code=400,
                error_code="TEST_ERROR"
            )

        assert isinstance(http_exception, HTTPException)
        assert http_exception.status_code == 400
        assert "テストエラー" in str(http_exception.detail)
        mock_logger.error.assert_called_once()

    def test_handle_model_error(self):
        """モデルエラーハンドリングテスト"""
        error = RuntimeError("モデルエラー")

        with patch('backend.app.utils.error_handler.logger') as mock_logger:
            response = ErrorHandler.handle_model_error(
                error,
                context="MLコンテキスト",
                operation="predict"
            )

        assert isinstance(response, dict)
        assert response["error_code"] == "MODEL_ERROR"
        assert "MLコンテキスト" in response.get("context", "")
        mock_logger.error.assert_called_once()

    def test_safe_execute_success(self):
        """安全実行成功テスト"""
        def test_func():
            return "success"

        result = ErrorHandler.safe_execute(test_func)
        assert result == "success"

    def test_safe_execute_with_exception(self):
        """安全実行例外テスト"""
        def test_func():
            raise ValueError("テストエラー")

        result = ErrorHandler.safe_execute(test_func, default_return="default")
        assert result == "default"

    def test_safe_execute_api_call_with_exception(self):
        """API呼び出し時の安全実行例外テスト"""
        def test_func():
            raise ValueError("APIエラー")

        with pytest.raises(HTTPException):
            ErrorHandler.safe_execute(
                test_func,
                is_api_call=True,
                api_status_code=400,
                api_error_code="API_ERROR"
            )

    @pytest.mark.asyncio
    async def test_safe_execute_async_success(self):
        """非同期安全実行成功テスト"""
        async def test_func():
            return "async_success"

        result = await ErrorHandler.safe_execute_async(test_func)
        assert result == "async_success"

    @pytest.mark.asyncio
    async def test_safe_execute_async_with_exception(self):
        """非同期安全実行例外テスト"""
        async def test_func():
            raise ValueError("非同期エラー")

        with pytest.raises(HTTPException):
            await ErrorHandler.safe_execute_async(test_func, status_code=400)

    @patch('backend.app.utils.error_handler.platform.system')
    @patch('backend.app.utils.error_handler.concurrent.futures.ThreadPoolExecutor')
    def test_handle_timeout_windows(self, mock_executor_class, mock_platform):
        """Windowsタイムアウト処理テスト"""
        mock_platform.return_value = "Windows"

        def test_func():
            return "success"

        result = ErrorHandler.handle_timeout(test_func, 5)
        assert result == "success"

    @patch('backend.app.utils.error_handler.platform.system')
    @patch('backend.app.utils.error_handler.signal')
    def test_handle_timeout_unix(self, mock_signal, mock_platform):
        """Unixタイムアウト処理テスト"""
        mock_platform.return_value = "Linux"

        def test_func():
            return "success"

        result = ErrorHandler.handle_timeout(test_func, 5)
        assert result == "success"

    @patch('backend.app.utils.error_handler.platform.system')
    @patch('backend.app.utils.error_handler.concurrent.futures.ThreadPoolExecutor')
    def test_handle_timeout_windows_timeout(self, mock_executor_class, mock_platform):
        """Windowsタイムアウト発生テスト"""
        mock_platform.return_value = "Windows"
        mock_executor = Mock()
        mock_future = Mock()
        mock_future.result.side_effect = TimeoutError("timeout")
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.submit.return_value = mock_future
        mock_executor_class.return_value = mock_executor

        def test_func():
            return "success"

        with pytest.raises(TimeoutError):
            ErrorHandler.handle_timeout(test_func, 5)

    def test_validate_predictions_valid(self):
        """有効な予測値バリデーションテスト"""
        predictions = {"up": 0.3, "down": 0.4, "range": 0.3}
        result = ErrorHandler.validate_predictions(predictions)
        assert result is True

    def test_validate_predictions_invalid_nan(self):
        """NaNを含む予測値バリデーションテスト"""
        predictions = {"up": 0.3, "down": float('nan'), "range": 0.3}
        result = ErrorHandler.validate_predictions(predictions)
        assert result is False

    def test_validate_predictions_invalid_inf(self):
        """Infを含む予測値バリデーションテスト"""
        predictions = {"up": 0.3, "down": float('inf'), "range": 0.3}
        result = ErrorHandler.validate_predictions(predictions)
        assert result is False

    def test_validate_dataframe_valid(self):
        """有効なデータフレームバリデーションテスト"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        result = ErrorHandler.validate_dataframe(
            df,
            required_columns=['col1', 'col2'],
            min_rows=2
        )
        assert result is True

    def test_validate_dataframe_empty(self):
        """空データフレームバリデーションテスト"""
        df = pd.DataFrame()
        result = ErrorHandler.validate_dataframe(df)
        assert result is False

    def test_validate_dataframe_missing_columns(self):
        """必須カラム不足データフレームバリデーションテスト"""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        result = ErrorHandler.validate_dataframe(
            df,
            required_columns=['col1', 'col2']
        )
        assert result is False

    def test_validate_dataframe_insufficient_rows(self):
        """行数不足データフレームバリデーションテスト"""
        df = pd.DataFrame({'col1': [1]})
        result = ErrorHandler.validate_dataframe(df, min_rows=5)
        assert result is False


class TestDecoratorsAndContexts:
    """デコレータとコンテキストマネージャーのテスト"""

    def test_timeout_decorator_success(self):
        """タイムアウトデコレータ成功テスト"""
        @timeout_decorator(5)
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_safe_operation_decorator_success(self):
        """安全操作デコレータ成功テスト"""
        @safe_operation(default_return="default")
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_safe_operation_decorator_with_exception(self):
        """安全操作デコレータ例外テスト"""
        @safe_operation(default_return="default")
        def test_func():
            raise ValueError("test error")

        result = test_func()
        assert result == "default"

    def test_safe_operation_decorator_raise_exception(self):
        """安全操作デコレータ例外再送テスト"""
        @safe_operation(default_return="RAISE_EXCEPTION")
        def test_func():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            test_func()

    def test_safe_ml_operation_alias(self):
        """safe_ml_operationエイリアステスト"""
        @safe_ml_operation(default_return="default")
        def test_func():
            raise ValueError("test error")

        result = test_func()
        assert result == "default"

    def test_operation_context_success(self):
        """操作コンテキスト成功テスト"""
        with patch('backend.app.utils.error_handler.logger') as mock_logger:
            with operation_context("テスト操作"):
                pass

        assert mock_logger.info.called
        # 開始と完了のログが呼ばれることを確認
        assert mock_logger.info.call_count >= 2

    def test_operation_context_with_exception(self):
        """操作コンテキスト例外テスト"""
        with patch('backend.app.utils.error_handler.logger') as mock_logger:
            with pytest.raises(ValueError):
                with operation_context("テスト操作"):
                    raise ValueError("test error")

        mock_logger.error.assert_called_once()


class TestCustomExceptions:
    """カスタム例外クラスのテスト"""

    def test_timeout_error(self):
        """TimeoutErrorテスト"""
        error = TimeoutError("タイムアウトしました")
        assert str(error) == "タイムアウトしました"
        assert isinstance(error, Exception)

    def test_validation_error(self):
        """ValidationErrorテスト"""
        error = ValidationError("バリデーションエラー")
        assert str(error) == "バリデーションエラー"
        assert isinstance(error, Exception)

    def test_data_error(self):
        """DataErrorテスト"""
        error = DataError("データエラー")
        assert str(error) == "データエラー"
        assert isinstance(error, Exception)

    def test_model_error(self):
        """ModelErrorテスト"""
        error = ModelError("モデルエラー")
        assert str(error) == "モデルエラー"
        assert isinstance(error, Exception)