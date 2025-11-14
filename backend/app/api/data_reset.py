"""
データリセットAPI

OHLCV、ファンディングレート、オープンインタレストデータのリセット機能を提供します。
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_data_management_orchestration_service
from app.services.data_collection.orchestration.data_management_orchestration_service import (
    DataManagementOrchestrationService,
)
from app.utils.error_handler import ErrorHandler
from database.connection import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data-reset", tags=["data-reset"])


@router.delete("/all")
async def reset_all_data(
    db: Session = Depends(get_db),
    orchestration_service: DataManagementOrchestrationService = Depends(
        get_data_management_orchestration_service
    ),
) -> Dict[str, Any]:
    """
    全てのデータ（OHLCV、ファンディングレート、オープンインタレスト）をリセット

    Returns:
        削除結果の詳細
    """

    async def _reset_all_data():
        return await orchestration_service.reset_all_data(db_session=db)

    return await ErrorHandler.safe_execute_async(
        _reset_all_data, message="全データのリセット中にエラーが発生しました"
    )


@router.delete("/ohlcv")
async def reset_ohlcv_data(
    db: Session = Depends(get_db),
    orchestration_service: DataManagementOrchestrationService = Depends(
        get_data_management_orchestration_service
    ),
) -> Dict[str, Any]:
    """
    OHLCVデータのみをリセット

    Returns:
        削除結果の詳細
    """

    async def _reset_ohlcv_data():
        return await orchestration_service.reset_ohlcv_data(db_session=db)

    return await ErrorHandler.safe_execute_async(
        _reset_ohlcv_data, message="OHLCVデータのリセット中にエラーが発生しました"
    )


@router.delete("/funding-rates")
async def reset_funding_rate_data(
    db: Session = Depends(get_db),
    orchestration_service: DataManagementOrchestrationService = Depends(
        get_data_management_orchestration_service
    ),
) -> Dict[str, Any]:
    """
    ファンディングレートデータのみをリセット

    Returns:
        削除結果の詳細
    """

    async def _reset_funding_rate_data():
        return await orchestration_service.reset_funding_rate_data(db_session=db)

    return await ErrorHandler.safe_execute_async(
        _reset_funding_rate_data,
        message="ファンディングレートデータのリセット中にエラーが発生しました",
    )


@router.delete("/open-interest")
async def reset_open_interest_data(
    db: Session = Depends(get_db),
    orchestration_service: DataManagementOrchestrationService = Depends(
        get_data_management_orchestration_service
    ),
) -> Dict[str, Any]:
    """
    オープンインタレストデータのみをリセット

    Returns:
        削除結果の詳細
    """

    async def _reset_open_interest():
        return await orchestration_service.reset_open_interest_data(db_session=db)

    return await ErrorHandler.safe_execute_async(
        _reset_open_interest,
        message="オープンインタレストデータのリセット中にエラーが発生しました",
    )


@router.delete("/symbol/{symbol:path}")
async def reset_data_by_symbol(
    symbol: str,
    db: Session = Depends(get_db),
    orchestration_service: DataManagementOrchestrationService = Depends(
        get_data_management_orchestration_service
    ),
) -> Dict[str, Any]:
    """
    特定シンボルの全データ（OHLCV、ファンディングレート、オープンインタレスト）をリセット

    Args:
        symbol: 削除対象のシンボル（例: BTC/USDT:USDT）

    Returns:
        削除結果の詳細
    """

    async def _reset_by_symbol():
        return await orchestration_service.reset_data_by_symbol(
            symbol=symbol, db_session=db
        )

    return await ErrorHandler.safe_execute_async(
        _reset_by_symbol, message="シンボル別データのリセット中にエラーが発生しました"
    )


@router.get("/status")
async def get_data_status(
    db: Session = Depends(get_db),
    orchestration_service: DataManagementOrchestrationService = Depends(
        get_data_management_orchestration_service
    ),
) -> Dict[str, Any]:
    """
    現在のデータ状況を取得（詳細版）

    Returns:
        各データタイプの詳細情報（件数、最新・最古データ）
    """

    async def _get_status():
        return await orchestration_service.get_data_status(db_session=db)

    return await ErrorHandler.safe_execute_async(
        _get_status, message="データステータスの取得中にエラーが発生しました"
    )
