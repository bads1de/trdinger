import pytest
import pandas as pd
from fastapi import HTTPException
from app.utils.response import api_response, error_response
from app.utils.error_handler import ErrorHandler, safe_operation, operation_context

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

    def test_error_handler_api(self):
        # APIエラーハンドリングの検証
        error = ValueError("Test API Error")
        http_exc = ErrorHandler.handle_api_error(error, context="APIContext", status_code=400)
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
        async def success(): return "ok"
        assert await ErrorHandler.safe_execute_async(success) == "ok"
        # 2. 失敗時 (HTTPException)
        async def fail(): raise HTTPException(status_code=400, detail="bad")
        with pytest.raises(HTTPException):
            await ErrorHandler.safe_execute_async(fail)

    def test_validate_predictions(self):
        # 正常系
        valid = {"up": 0.4, "down": 0.4, "range": 0.2} # 合計 1.0
        assert ErrorHandler.validate_predictions(valid) is True
        # 異常系: キー不足
        assert ErrorHandler.validate_predictions({"up": 0.5}) is False
        # 異常系: 値が範囲外
        assert ErrorHandler.validate_predictions({"up": 1.5, "down": -0.1, "range": 0.1}) is False
        # 異常系: 合計が範囲外 (0.8 - 1.2 外)
        assert ErrorHandler.validate_predictions({"up": 0.1, "down": 0.1, "range": 0.1}) is False

    def test_validate_dataframe(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        # 正常
        assert ErrorHandler.validate_dataframe(df, required_columns=["A"], min_rows=2) is True
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

