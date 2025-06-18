from fastapi import HTTPException
import logging
from typing import Callable, Awaitable, Any, Optional

logger = logging.getLogger(__name__)


async def handle_api_exception(
    call: Callable[..., Awaitable[Any]],
    message: str = "Internal Server Error",
    status_code: int = 500,
):
    """
    APIエンドポイントで発生した例外を処理し、HTTPExceptionを発生させるヘルパー関数。
    """
    try:
        return await call()
    except HTTPException as e:
        logger.error(f"{message}: {e.detail}", exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"{message}: {e}", exc_info=True)
        raise HTTPException(status_code=status_code, detail=message)


def log_exception(e: Exception, message: str = "An unexpected error occurred"):
    """
    例外をログに記録するためのヘルパー関数。HTTPExceptionを発生させない。
    """
    logger.error(f"{message}: {e}", exc_info=True)


def api_response(
    success: bool,
    message: str,
    status: Optional[str] = None,
    data: Optional[dict] = None,
):
    """
    標準化されたAPIレスポンスを生成するヘルパー関数。
    """
    response = {"success": success, "message": message}
    if status:
        response["status"] = status
    if data:
        response["data"] = data
    return response
