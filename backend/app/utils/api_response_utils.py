from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)


def handle_api_exception(
    e: Exception, message: str = "Internal Server Error", status_code: int = 500
):
    """
    APIエンドポイントで発生した例外を処理し、HTTPExceptionを発生させるヘルパー関数。
    """
    logger.error(f"{message}: {e}", exc_info=True)
    if isinstance(e, HTTPException):
        raise e
    raise HTTPException(status_code=status_code, detail=message)


def log_exception(e: Exception, message: str = "An unexpected error occurred"):
    """
    例外をログに記録するためのヘルパー関数。HTTPExceptionを発生させない。
    """
    logger.error(f"{message}: {e}", exc_info=True)


def api_response(success: bool, message: str, status: str = None, data: dict = None):
    """
    標準化されたAPIレスポンスを生成するヘルパー関数。
    """
    response = {"success": success, "message": message}
    if status:
        response["status"] = status
    if data:
        response["data"] = data
    return response
