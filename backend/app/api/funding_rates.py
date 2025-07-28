"""
ファンディングレートAPI

ファンディングレートデータの取得・収集機能を提供するAPIエンドポイント
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional


from database.connection import get_db, ensure_db_initialized
from app.utils.unified_error_handler import UnifiedErrorHandler
from app.services.data_collection.orchestration.funding_rate_orchestration_service import (
    FundingRateOrchestrationService,
)
from app.api.dependencies import get_funding_rate_orchestration_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/funding-rates", tags=["funding-rates"])


@router.get("/")
async def get_funding_rates(
    symbol: str = Query(..., description="取引ペアシンボル（例: 'BTC/USDT'）"),
    limit: Optional[int] = Query(100, description="取得するデータ数（1-1000）"),
    start_date: Optional[str] = Query(None, description="開始日時（ISO形式）"),
    end_date: Optional[str] = Query(None, description="終了日時（ISO形式）"),
    orchestration_service: FundingRateOrchestrationService = Depends(
        get_funding_rate_orchestration_service
    ),
    db: Session = Depends(get_db),
):
    """
    ファンディングレートデータを取得します

    データベースに保存されたファンディングレートデータを取得します。

    Args:
        symbol: 取引ペアシンボル（例: 'BTC/USDT'）
        limit: 取得するデータ数（1-1000）
        start_date: 開始日時（ISO形式）
        end_date: 終了日時（ISO形式）
        orchestration_service: ファンディングレートサービス（依存性注入）
        db: データベースセッション

    Returns:
        ファンディングレートデータを含むJSONレスポンス

    Raises:
        HTTPException: パラメータが無効な場合やデータベースエラーが発生した場合
    """

    async def _get_funding_rates_data():
        return await orchestration_service.get_funding_rate_data(
            symbol=symbol,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            db_session=db,
        )

    async def _get_funding_rates_data():
        return await orchestration_service.get_funding_rate_data(
            symbol=symbol,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            db_session=db,
        )

    return await UnifiedErrorHandler.safe_execute_async(
        _get_funding_rates_data, message="ファンディングレートデータ取得エラー"
    )


@router.post("/collect")
async def collect_funding_rate_data(
    symbol: str = Query(..., description="取引ペアシンボル（例: 'BTC/USDT'）"),
    limit: Optional[int] = Query(
        100, description="取得するデータ数（1-1000、fetch_all=trueの場合は無視）"
    ),
    fetch_all: bool = Query(False, description="全期間のデータを取得するかどうか"),
    orchestration_service: FundingRateOrchestrationService = Depends(
        get_funding_rate_orchestration_service
    ),
    db: Session = Depends(get_db),
):
    """
    ファンディングレートデータを収集してデータベースに保存します

    取引所からファンディングレートデータを取得し、データベースに保存します。

    Args:
        symbol: 取引ペアシンボル（例: 'BTC/USDT'）
        limit: 取得するデータ数（1-1000、fetch_all=trueの場合は無視）
        fetch_all: 全期間のデータを取得するかどうか
        db: データベースセッション

    Returns:
        収集結果を含むJSONレスポンス

    Raises:
        HTTPException: パラメータが無効な場合やAPI/データベースエラーが発生した場合
    """

    async def _collect_rates():
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました")
            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        return await orchestration_service.collect_funding_rate_data(
            symbol=symbol,
            limit=limit,
            fetch_all=fetch_all,
            db_session=db,
        )

    return await UnifiedErrorHandler.safe_execute_async(
        _collect_rates, message="ファンディングレートデータ収集エラー"
    )


@router.post("/bulk-collect")
async def bulk_collect_funding_rates(
    orchestration_service: FundingRateOrchestrationService = Depends(
        get_funding_rate_orchestration_service
    ),
    db: Session = Depends(get_db),
):
    """
    BTCシンボルのファンディングレートデータを一括収集します

    BTCの無期限契約シンボルの全期間ファンディングレートデータを一括で取得・保存します。
    ETHは分析対象から除外されています。

    Args:
        db: データベースセッション

    Returns:
        一括収集結果を含むJSONレスポンス

    Raises:
        HTTPException: データベースエラーが発生した場合
    """

    async def _bulk_collect():
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました")
            raise Exception("データベースの初期化に失敗しました")

        symbols = [
            "BTC/USDT:USDT",
        ]

        return await orchestration_service.collect_bulk_funding_rate_data(
            symbols=symbols, db_session=db
        )

    return await UnifiedErrorHandler.safe_execute_async(
        _bulk_collect, message="ファンディングレート一括収集エラー"
    )
