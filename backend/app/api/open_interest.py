"""
オープンインタレストAPIエンドポイント

ファンディングレートAPIの実装パターンを参考に、
オープンインタレスト（建玉残高）データの取得・収集機能を提供します。
"""

import logging

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from database.connection import get_db, ensure_db_initialized
from app.utils.unified_error_handler import UnifiedErrorHandler
from app.services.data_collection.orchestration.open_interest_orchestration_service import (
    OpenInterestOrchestrationService,
)
from app.api.dependencies import get_open_interest_orchestration_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/open-interest", tags=["open-interest"])


@router.get("/")
async def get_open_interest_data(
    symbol: str = Query(..., description="取引ペアシンボル（例: 'BTC/USDT'）"),
    start_date: Optional[str] = Query(None, description="開始日時（ISO形式）"),
    end_date: Optional[str] = Query(None, description="終了日時（ISO形式）"),
    limit: Optional[int] = Query(1000, description="取得件数制限（最大1000）"),
    orchestration_service: OpenInterestOrchestrationService = Depends(
        get_open_interest_orchestration_service
    ),
    db: Session = Depends(get_db),
):
    """
    オープンインタレストデータを取得します
    """

    async def _get_data():
        return await orchestration_service.get_open_interest_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            db_session=db,
        )

    return await UnifiedErrorHandler.safe_execute_async(
        _get_data, message="オープンインタレストデータ取得エラー"
    )


@router.post("/collect")
async def collect_open_interest_data(
    symbol: str = Query(..., description="取引ペアシンボル（例: 'BTC/USDT'）"),
    limit: Optional[int] = Query(
        100, description="取得するデータ数（1-1000、fetch_all=trueの場合は無視）"
    ),
    fetch_all: bool = Query(False, description="全期間のデータを取得するかどうか"),
    orchestration_service: OpenInterestOrchestrationService = Depends(
        get_open_interest_orchestration_service
    ),
    db: Session = Depends(get_db),
):
    """
    オープンインタレストデータを収集してデータベースに保存します

    Bybit取引所からオープンインタレストデータを取得し、データベースに保存します。

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

    async def _collect_open_interest():
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました")
            raise Exception("データベースの初期化に失敗しました")

        return await orchestration_service.collect_open_interest_data(
            symbol=symbol,
            limit=limit,
            fetch_all=fetch_all,
            db_session=db,
        )

    return await UnifiedErrorHandler.safe_execute_async(
        _collect_open_interest, message="オープンインタレストデータ収集エラー"
    )


@router.post("/bulk-collect")
async def bulk_collect_open_interest(
    orchestration_service: OpenInterestOrchestrationService = Depends(
        get_open_interest_orchestration_service
    ),
    db: Session = Depends(get_db),
):
    """
    BTCシンボルのオープンインタレストデータを一括収集します

    BTCの無期限契約シンボルの全期間オープンインタレストデータを一括で取得・保存します。
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

        return await orchestration_service.collect_bulk_open_interest_data(
            symbols=symbols, db_session=db
        )

    return await UnifiedErrorHandler.safe_execute_async(
        _bulk_collect, message="オープンインタレスト一括収集エラー"
    )
