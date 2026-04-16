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
@ErrorHandler.api_endpoint("全データのリセット中にエラーが発生しました")
async def reset_all_data(
    db: Session = Depends(get_db),
    orchestration_service: DataManagementOrchestrationService = Depends(
        get_data_management_orchestration_service
    ),
) -> Dict[str, Any]:
    """全ての市場データ（OHLCV、FR、OI）をリセットする。

    データベースから全てのOHLCVデータ、ファンディングレート、
    オープンインタレストデータを削除します。

    Args:
        db: データベースセッション（依存性注入）。
        orchestration_service: データ管理オーケストレーションサービス（依存性注入）。

    Returns:
        Dict[str, Any]: 削除結果の詳細。成功した場合は削除対象の件数を含む。
    """
    return await orchestration_service.reset_all_data(db_session=db)


@router.delete("/ohlcv")
@ErrorHandler.api_endpoint("OHLCVデータのリセット中にエラーが発生しました")
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
    return await orchestration_service.reset_ohlcv_data(db_session=db)


@router.delete("/funding-rates")
@ErrorHandler.api_endpoint(
    "ファンディングレートデータのリセット中にエラーが発生しました"
)
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
    return await orchestration_service.reset_funding_rate_data(db_session=db)


@router.delete("/open-interest")
@ErrorHandler.api_endpoint(
    "オープンインタレストデータのリセット中にエラーが発生しました"
)
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
    return await orchestration_service.reset_open_interest_data(db_session=db)


@router.delete("/symbol/{symbol:path}")
@ErrorHandler.api_endpoint("シンボル別データのリセット中にエラーが発生しました")
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
    return await orchestration_service.reset_data_by_symbol(
        symbol=symbol, db_session=db
    )


@router.get("/status")
@ErrorHandler.api_endpoint("データステータスの取得中にエラーが発生しました")
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
    return await orchestration_service.get_data_status(db_session=db)
