import pytest
import pandas as pd
from fastapi import HTTPException

from app.utils.error_handler import ErrorHandler, ensure_db_initialized, safe_operation
from app.utils.response import (
    api_response,
    ensure_response_dict,
    error_response,
    extract_response_data,
    result_response,
)


class TestUtilsExtra:
    def test_api_response(self):
        # 正常なレスポンス
        resp = api_response(success=True, data={"key": "value"}, message="Success")
        assert resp["success"] is True
        assert resp["data"] == {"key": "value"}
        assert resp["message"] == "Success"
        assert "timestamp" in resp

    def test_error_response(self):
        resp = error_response(message="Failed", error_code="ERR001", context="Test")
        assert resp["success"] is False
        assert resp["message"] == "Failed"
        assert resp["error_code"] == "ERR001"

    def test_result_response(self):
        success_resp = result_response(
            success=True,
            message="OK",
            data={"key": "value"},
        )
        error_resp = result_response(
            success=False,
            message="Failed",
            error_code="ERR001",
            details={"reason": "bad"},
            data={},
        )

        assert success_resp["success"] is True
        assert success_resp["data"] == {"key": "value"}
        assert error_resp["success"] is False
        assert error_resp["error_code"] == "ERR001"
        assert error_resp["details"] == {"reason": "bad"}

    def test_ensure_response_dict(self):
        class ModelDumpResult:
            def model_dump(self):
                return {"success": True, "data": {"key": "value"}}

        class DictResult:
            def dict(self):
                return {"success": False, "message": "fallback"}

        assert ensure_response_dict({"success": True}) == {"success": True}
        assert ensure_response_dict(ModelDumpResult()) == {
            "success": True,
            "data": {"key": "value"},
        }
        assert ensure_response_dict(DictResult()) == {
            "success": False,
            "message": "fallback",
        }
        assert ensure_response_dict(object()) == {}

    def test_extract_response_data(self):
        assert extract_response_data({"data": {"key": "value"}}) == {
            "key": "value"
        }
        assert extract_response_data({"data": None}) == {}

    def test_ensure_db_initialized_success(self, monkeypatch):
        monkeypatch.setattr("database.connection.init_db", lambda: True)

        ensure_db_initialized()

    def test_ensure_db_initialized_failure(self, monkeypatch):
        monkeypatch.setattr("database.connection.init_db", lambda: False)

        with pytest.raises(HTTPException) as exc_info:
            ensure_db_initialized()

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "データベースの初期化に失敗しました"

    def test_error_handler_api(self):
        # APIエラーハンドリングの検証
        error = ValueError("Test API Error")
        http_exc = ErrorHandler.handle_api_error(
            error, context="APIContext", status_code=400
        )
        assert http_exc.status_code == 400
        assert http_exc.detail["success"] is False
        assert "Test API Error" in http_exc.detail["message"]

    def test_safe_operation_decorator(self):
        # safe_operation デコレータの検証
        @safe_operation(default_return="fallback")
        def fail_func():
            raise ValueError("Failure")

        assert fail_func() == "fallback"

    @pytest.mark.asyncio
    async def test_safe_execute_async(self):
        # 1. 成功時
        async def success():
            return "ok"

        assert await ErrorHandler.safe_execute_async(success) == "ok"

        # 2. 失敗時 (HTTPException)
        async def fail():
            raise HTTPException(status_code=400, detail="bad")

        with pytest.raises(HTTPException):
            await ErrorHandler.safe_execute_async(fail)

    def test_validate_predictions(self):
        # 正常系
        valid = {"up": 0.4, "down": 0.4, "range": 0.2}  # 合計 1.0
        assert ErrorHandler.validate_predictions(valid) is True
        # 異常系: キー不足
        assert ErrorHandler.validate_predictions({"up": 0.5}) is False
        # 異常系: 値が範囲外
        assert (
            ErrorHandler.validate_predictions({"up": 1.5, "down": -0.1, "range": 0.1})
            is False
        )
        # 異常系: 合計が範囲外 (0.8 - 1.2 外)
        assert (
            ErrorHandler.validate_predictions({"up": 0.1, "down": 0.1, "range": 0.1})
            is False
        )

    def test_validate_dataframe(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        # 正常
        assert (
            ErrorHandler.validate_dataframe(df, required_columns=["A"], min_rows=2)
            is True
        )
        # 異常: 行数不足
        assert ErrorHandler.validate_dataframe(df, min_rows=10) is False
        # 異常: カラム不足
        assert ErrorHandler.validate_dataframe(df, required_columns=["C"]) is False
        # 異常: 空
        assert ErrorHandler.validate_dataframe(pd.DataFrame()) is False

    def test_handle_timeout_logic(self):
        # Windows/Unix分岐を含むタイムアウト処理の導通
        def slow_func():
            return "done"

        # タイムアウトせずに終了する場合
        assert ErrorHandler.handle_timeout(slow_func, 5) == "done"
