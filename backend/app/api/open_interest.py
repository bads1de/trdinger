"""
オープンインタレストAPIエンドポイント

ファンディングレートAPIの実装パターンを参考に、
オープンインタレスト（建玉残高）データの取得・収集機能を提供します。
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.dependencies import get_open_interest_orchestration_service
from app.config.constants import DEFAULT_MARKET_SYMBOL
from app.services.data_collection.orchestration.open_interest_orchestration_service import (  # noqa: E501
    OpenInterestOrchestrationService,
)
from app.utils.error_handler import api_safe_execute, ensure_db_initialized
from database.connection import get_db

router = APIRouter(
    prefix="/api/open-interest",
    tags=["open-interest"],
    dependencies=[Depends(ensure_db_initialized)],
)


@router.get("/")
@api_safe_execute(message="オープンインタレストデータ取得エラー")
async def get_open_interest_data(
    symbol: str = Query(
        ..., description="取引ペアシンボル（例: 'BTC/USDT:USDT'）"
    ),
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

    データベースに保存されたオープンインタレスト（建玉残高）データを取得します。
    期間指定または件数制限でデータをフィルタリングできます。

    Args:
        symbol: 取引ペアシンボル（例: 'BTC/USDT:USDT'）
        start_date: 取得開始日時（ISO形式）
        end_date: 取得終了日時（ISO形式）
        limit: 取得件数制限（1-1000）
        orchestration_service: オープンインタレストサービス（依存性注入）
        db: データベースセッション

    Returns:
        オープンインタレストデータを含むJSONレスポンス

    Raises:
        HTTPException: パラメータが無効な場合やデータベースエラーが発生した場合
    """
    return await orchestration_service.get_open_interest_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        db_session=db,
    )


@router.post("/collect")
@api_safe_execute(message="オープンインタレストデータ収集エラー")
async def collect_open_interest_data(
    symbol: str = Query(
        ..., description="取引ペアシンボル（例: 'BTC/USDT:USDT'）"
    ),
    limit: Optional[int] = Query(
        100,
        description="取得するデータ数（1-1000、fetch_all=trueの場合は無視）",
    ),
    fetch_all: bool = Query(
        False, description="全期間のデータを取得するかどうか"
    ),
    orchestration_service: OpenInterestOrchestrationService = Depends(
        get_open_interest_orchestration_service
    ),
    db: Session = Depends(get_db),
):
    """
    オープンインタレストデータを収集してデータベースに保存します

    Bybit取引所からオープンインタレストデータを取得し、データベースに保存します。

    Args:
        symbol: 取引ペアシンボル（例: 'BTC/USDT:USDT'）
        limit: 取得するデータ数（1-1000、fetch_all=trueの場合は無視）
        fetch_all: 全期間のデータを取得するかどうか
        db: データベースセッション

    Returns:
        収集結果を含むJSONレスポンス

    Raises:
        HTTPException: パラメータが無効な場合やAPI/データベースエラーが発生した場合
    """
    return await orchestration_service.collect_open_interest_data(
        symbol=symbol,
        limit=limit,
        fetch_all=fetch_all,
        db_session=db,
    )


@router.post("/bulk-collect")
@api_safe_execute(message="オープンインタレスト一括収集エラー")
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
    symbols = [
        DEFAULT_MARKET_SYMBOL,
    ]

    return await orchestration_service.collect_bulk_open_interest_data(
        symbols=symbols, db_session=db
    )
