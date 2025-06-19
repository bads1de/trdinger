"""
API共通ユーティリティ
"""

from fastapi import HTTPException
from datetime import datetime
from typing import Any, Dict, Optional, Callable, Awaitable
import logging

logger = logging.getLogger(__name__)


class APIResponseHelper:
    """API レスポンス形式の共通ヘルパークラス"""

    @staticmethod
    def success_response(
        data: Any, message: str, additional_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        成功レスポンスを生成

        Args:
            data: レスポンスデータ
            message: メッセージ
            additional_fields: 追加フィールド

        Returns:
            成功レスポンス辞書
        """
        response = {
            "success": True,
            "data": data,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }

        if additional_fields:
            response.update(additional_fields)

        return response

    @staticmethod
    def error_response(
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        エラーレスポンスを生成

        Args:
            message: エラーメッセージ
            error_code: エラーコード
            details: エラー詳細

        Returns:
            エラーレスポンス辞書
        """
        response = {
            "success": False,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }

        if error_code:
            response["error_code"] = error_code

        if details:
            response["details"] = details

        return response

    @staticmethod
    def api_response(
        success: bool,
        message: str,
        status: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        標準化されたAPIレスポンスを生成するヘルパー関数。
        """
        response = {"success": success, "message": message}
        if status:
            response["status"] = status
        if data:
            response["data"] = data
        response["timestamp"] = (
            datetime.now().isoformat()
        )  # Add timestamp for consistency
        return response


class APIErrorHandler:
    """API エラーハンドリングの共通ヘルパークラス"""

    @staticmethod
    def handle_validation_error(error: Exception, context: str = "") -> HTTPException:
        """
        バリデーションエラーを処理

        Args:
            error: 発生したエラー
            context: エラーコンテキスト

        Returns:
            HTTPException
        """
        error_message = f"バリデーションエラー: {str(error)}"
        if context:
            error_message = f"{context} - {error_message}"

        logger.error(error_message)

        return HTTPException(
            status_code=400,
            detail=APIResponseHelper.error_response(
                message=str(error), error_code="VALIDATION_ERROR"
            ),
        )

    @staticmethod
    def handle_database_error(error: Exception, context: str = "") -> HTTPException:
        """
        データベースエラーを処理

        Args:
            error: 発生したエラー
            context: エラーコンテキスト

        Returns:
            HTTPException
        """
        error_message = f"データベースエラー: {str(error)}"
        if context:
            error_message = f"{context} - {error_message}"

        logger.error(error_message)

        return HTTPException(
            status_code=500,
            detail=APIResponseHelper.error_response(
                message="データベースエラーが発生しました", error_code="DATABASE_ERROR"
            ),
        )

    @staticmethod
    def handle_external_api_error(error: Exception, context: str = "") -> HTTPException:
        """
        外部APIエラーを処理

        Args:
            error: 発生したエラー
            context: エラーコンテキスト

        Returns:
            HTTPException
        """
        error_message = f"外部APIエラー: {str(error)}"
        if context:
            error_message = f"{context} - {error_message}"

        logger.error(error_message)

        return HTTPException(
            status_code=502,
            detail=APIResponseHelper.error_response(
                message="外部APIとの通信でエラーが発生しました",
                error_code="EXTERNAL_API_ERROR",
            ),
        )

    @staticmethod
    def handle_generic_error(error: Exception, context: str = "") -> HTTPException:
        """
        一般的なエラーを処理

        Args:
            error: 発生したエラー
            context: エラーコンテキスト

        Returns:
            HTTPException
        """
        error_message = f"予期しないエラー: {str(error)}"
        if context:
            error_message = f"{context} - {error_message}"

        logger.error(error_message)

        return HTTPException(
            status_code=500,
            detail=APIResponseHelper.error_response(
                message="予期しないエラーが発生しました", error_code="INTERNAL_ERROR"
            ),
        )

    @staticmethod
    async def handle_api_exception(
        call: Callable[..., Awaitable[Any]],
        message: str = "Internal Server Error",
        status_code: int = 500,
    ) -> Any:
        """
        APIエンドポイントで発生した例外を処理し、HTTPExceptionを発生させるヘルパー関数。
        """
        try:
            return await call()
        except HTTPException as e:
            logger.error(f"API例外処理: {message} - {e.detail}", exc_info=True)
            raise e
        except Exception as e:
            logger.error(f"API例外処理: {message} - {e}", exc_info=True)
            raise HTTPException(status_code=status_code, detail=message)


class DateTimeHelper:
    """日時処理の共通ヘルパークラス"""

    @staticmethod
    def parse_iso_datetime(date_string: str) -> datetime:
        """
        ISO形式の日時文字列をdatetimeオブジェクトに変換

        Args:
            date_string: ISO形式の日時文字列

        Returns:
            datetimeオブジェクト

        Raises:
            ValueError: 日時形式が無効な場合
        """
        try:
            # "Z"をタイムゾーン情報に変換
            normalized_string = date_string.replace("Z", "+00:00")
            return datetime.fromisoformat(normalized_string)
        except ValueError as e:
            raise ValueError(f"無効な日時形式が指定されました: {date_string}") from e

    @staticmethod
    def timestamp_to_datetime(timestamp_ms: int) -> datetime:
        """
        ミリ秒タイムスタンプをdatetimeオブジェクトに変換

        Args:
            timestamp_ms: ミリ秒タイムスタンプ

        Returns:
            datetimeオブジェクト
        """
        from datetime import timezone

        return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)

    @staticmethod
    def datetime_to_timestamp(dt: datetime) -> int:
        """
        datetimeオブジェクトをミリ秒タイムスタンプに変換

        Args:
            dt: datetimeオブジェクト

        Returns:
            ミリ秒タイムスタンプ
        """
        return int(dt.timestamp() * 1000)
