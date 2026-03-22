"""API 層で共通利用する小さなヘルパー。"""

import logging

from fastapi import HTTPException, status

from database.connection import init_db

logger = logging.getLogger(__name__)

DEFAULT_DB_INIT_ERROR_MESSAGE = "データベースの初期化に失敗しました"


def ensure_db_initialized(error_message: str = DEFAULT_DB_INIT_ERROR_MESSAGE) -> None:
    """DB が利用可能でなければ 500 を返す。"""
    if init_db():
        return

    logger.error(error_message)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=error_message,
    )
