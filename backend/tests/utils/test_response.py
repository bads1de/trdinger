"""
Responseユーティリティのユニットテスト

APIレスポンス生成関数のテストモジュール
"""

from datetime import datetime
from typing import Any, Dict
from unittest.mock import patch

import pytest

from app.utils.response import (
    _build_response,
    api_response,
    ensure_response_dict,
    error_response,
    extract_response_data,
    now_iso,
    result_response,
)


class TestNowIso:
    """now_iso関数のテスト"""

    def test_returns_iso_format_string(self):
        """ISO8601形式の文字列を返す"""
        result = now_iso()
        assert isinstance(result, str)
        # ISO8601形式の検証（YYYY-MM-DDTHH:MM:SS形式）
        assert "T" in result

    def test_returns_current_time(self):
        """現在時刻を返す"""
        before = datetime.now()
        result = now_iso()
        after = datetime.now()

        result_dt = datetime.fromisoformat(result)
        assert before <= result_dt <= after


class TestEnsureResponseDict:
    """ensure_response_dict関数のテスト"""

    def test_returns_dict_unchanged(self):
        """辞書の場合はそのまま返す"""
        input_dict = {"key": "value", "number": 123}
        result = ensure_response_dict(input_dict)
        assert result == input_dict
        assert result is input_dict

    def test_extracts_from_pydantic_model(self):
        """Pydanticモデルの場合はmodel_dump()を呼び出す"""
        class MockModel:
            def model_dump(self):
                return {"name": "test", "value": 42}

        model = MockModel()
        result = ensure_response_dict(model)
        assert result == {"name": "test", "value": 42}

    def test_extracts_from_dict_method(self):
        """dict()メソッドを持つオブジェクトの場合は呼び出す"""
        class OldStyleModel:
            def dict(self):
                return {"legacy": True}

        model = OldStyleModel()
        result = ensure_response_dict(model)
        assert result == {"legacy": True}

    def test_returns_empty_dict_for_other_types(self):
        """その他の型の場合は空辞書を返す"""
        assert ensure_response_dict("string") == {}
        assert ensure_response_dict(123) == {}
        assert ensure_response_dict(None) == {}
        assert ensure_response_dict([1, 2, 3]) == {}

    def test_prefers_model_dump_over_dict(self):
        """model_dump()が存在する場合はdict()より優先する"""
        class BothMethods:
            def model_dump(self):
                return {"method": "model_dump"}

            def dict(self):
                return {"method": "dict"}

        model = BothMethods()
        result = ensure_response_dict(model)
        assert result == {"method": "model_dump"}

    def test_handles_model_dump_returning_non_dict(self):
        """model_dump()が辞書以外を返す場合は次の方法を試す"""
        class BadModel:
            def model_dump(self):
                return "not a dict"

            def dict(self):
                return {"fallback": True}

        model = BadModel()
        result = ensure_response_dict(model)
        assert result == {"fallback": True}


class TestExtractResponseData:
    """extract_response_data関数のテスト"""

    def test_extracts_data_by_key(self):
        """指定されたキーのデータを抽出する"""
        response = {"data": {"items": [1, 2, 3]}, "status": "ok"}
        result = extract_response_data(response, "data")
        assert result == {"items": [1, 2, 3]}

    def test_returns_empty_dict_if_key_missing(self):
        """キーが存在しない場合は空辞書を返す"""
        response = {"status": "ok"}
        result = extract_response_data(response, "data")
        assert result == {}

    def test_returns_empty_dict_if_value_not_dict(self):
        """値が辞書でない場合は空辞書を返す"""
        response = {"data": "not a dict"}
        result = extract_response_data(response, "data")
        assert result == {}

    def test_custom_key(self):
        """カスタムキーを使用できる"""
        response = {"payload": {"content": "test"}, "meta": {}}
        result = extract_response_data(response, "payload")
        assert result == {"content": "test"}


class TestBuildResponse:
    """_build_response関数のテスト"""

    def test_builds_success_response(self):
        """成功レスポンスを構築する"""
        result = _build_response(True, {"message": "ok", "data": {"id": 1}})

        assert result["success"] is True
        assert result["message"] == "ok"
        assert result["data"] == {"id": 1}
        assert "timestamp" in result

    def test_builds_error_response(self):
        """エラーレスポンスを構築する"""
        result = _build_response(False, {"message": "error", "error_code": "E001"})

        assert result["success"] is False
        assert result["message"] == "error"
        assert result["error_code"] == "E001"
        assert "timestamp" in result

    def test_excludes_none_values(self):
        """None値のフィールドを除外する"""
        result = _build_response(True, {"message": "ok", "data": None})

        assert "data" not in result
        assert result["message"] == "ok"

    def test_excludes_empty_string_values(self):
        """空文字列のフィールドを除外する"""
        result = _build_response(True, {"message": "ok", "detail": ""})

        assert "detail" not in result
        assert result["message"] == "ok"

    def test_includes_timestamp(self):
        """timestampフィールドを含める"""
        with patch("app.utils.response.now_iso") as mock_now:
            mock_now.return_value = "2024-01-15T10:30:00"
            result = _build_response(True, {"message": "ok"})

        assert result["timestamp"] == "2024-01-15T10:30:00"

    def test_preserves_zero_and_false_values(self):
        """0やFalseなどのfalsyだが有効な値を保持する"""
        result = _build_response(True, {
            "count": 0,
            "active": False,
            "empty_list": [],
        })

        assert result["count"] == 0
        assert result["active"] is False
        assert result["empty_list"] == []


class TestErrorResponse:
    """error_response関数のテスト"""

    def test_basic_error_response(self):
        """基本的なエラーレスポンスを生成する"""
        result = error_response("Something went wrong")

        assert result["success"] is False
        assert result["message"] == "Something went wrong"
        assert "timestamp" in result

    def test_error_response_with_all_fields(self):
        """全フィールドを含むエラーレスポンスを生成する"""
        result = error_response(
            message="Validation failed",
            error_code="VAL001",
            details={"field": "email", "issue": "invalid format"},
            context="user_registration",
            data={"attempted_value": "test@"},
        )

        assert result["success"] is False
        assert result["message"] == "Validation failed"
        assert result["error_code"] == "VAL001"
        assert result["details"] == {"field": "email", "issue": "invalid format"}
        assert result["context"] == "user_registration"
        assert result["data"] == {"attempted_value": "test@"}

    def test_error_response_excludes_none(self):
        """Noneのオプションフィールドを除外する"""
        result = error_response("Simple error")

        assert "error_code" not in result
        assert "details" not in result
        assert "context" not in result


class TestApiResponse:
    """api_response関数のテスト"""

    def test_success_response(self):
        """成功レスポンスを生成する"""
        result = api_response(
            success=True,
            message="Operation completed",
            data={"id": 123},
        )

        assert result["success"] is True
        assert result["message"] == "Operation completed"
        assert result["data"] == {"id": 123}

    def test_error_response(self):
        """エラーレスポンスを生成する"""
        result = api_response(
            success=False,
            message="Operation failed",
            error="Internal error",
        )

        assert result["success"] is False
        assert result["message"] == "Operation failed"
        assert result["error"] == "Internal error"

    def test_response_with_status_code(self):
        """ステータスコードを含むレスポンスを生成する"""
        result = api_response(
            success=True,
            message="Created",
            status_code=201,
        )

        assert result["status_code"] == 201

    def test_response_with_status(self):
        """ステータス文字列を含むレスポンスを生成する"""
        result = api_response(
            success=True,
            message="Done",
            status="completed",
        )

        assert result["status"] == "completed"

    def test_allows_empty_message(self):
        """空のメッセージは除外される"""
        result = api_response(success=True, message="")

        # 空文字列は_build_responseで除外される
        assert "message" not in result


class TestResultResponse:
    """result_response関数のテスト"""

    def test_success_result(self):
        """成功結果を生成する"""
        result = result_response(
            success=True,
            message="Data retrieved",
            data={"users": []},
        )

        assert result["success"] is True
        assert result["message"] == "Data retrieved"
        assert result["data"] == {"users": []}

    def test_failure_result(self):
        """失敗結果を生成する"""
        result = result_response(
            success=False,
            message="Database error",
            error_code="DB001",
            details={"table": "users"},
        )

        assert result["success"] is False
        assert result["message"] == "Database error"
        assert result["error_code"] == "DB001"
        assert result["details"] == {"table": "users"}

    def test_failure_result_with_data(self):
        """データ付きの失敗結果を生成する"""
        result = result_response(
            success=False,
            message="Partial failure",
            error_code="PARTIAL",
            data={"completed_items": 5},
        )

        assert result["success"] is False
        assert result["data"] == {"completed_items": 5}

    def test_with_status_and_status_code(self):
        """ステータスとステータスコードを含む結果を生成する"""
        result = result_response(
            success=True,
            message="Processed",
            status="processing",
            status_code=202,
        )

        assert result["status"] == "processing"
        assert result["status_code"] == 202


class TestResponseIntegration:
    """レスポンス関数の統合テスト"""

    def test_error_response_consistency(self):
        """error_responseとapi_responseのエラーモードの一貫性"""
        error_result = error_response(
            message="Test error",
            error_code="TEST001",
            details={"info": "test"},
        )

        api_error_result = api_response(
            success=False,
            message="Test error",
        )

        assert error_result["success"] == api_error_result["success"]
        assert error_result["message"] == api_error_result["message"]

    def test_timestamp_format_consistency(self):
        """すべてのレスポンスでtimestamp形式が一貫している"""
        with patch("app.utils.response.now_iso") as mock_now:
            mock_now.return_value = "2024-01-15T10:30:00.000000"

            error_r = error_response("test")
            api_r = api_response(True, "test")
            result_r = result_response(True, "test")

            assert error_r["timestamp"] == api_r["timestamp"] == result_r["timestamp"]
