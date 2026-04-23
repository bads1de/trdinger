"""
ファンディングレートAPI

ファンディングレートデータの取得・収集機能を提供するAPIエンドポイント
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.dependencies import get_funding_rate_orchestration_service
from app.config.constants import DEFAULT_MARKET_SYMBOL
from app.services.data_collection.orchestration.funding_rate_orchestration_service import (  # noqa: E501
    FundingRateOrchestrationService,
)
from app.utils.error_handler import api_safe_execute, ensure_db_initialized
from database.connection import get_db

router = APIRouter(
    prefix="/api/funding-rates",
    tags=["funding-rates"],
    dependencies=[Depends(ensure_db_initialized)],
)


@router.get("/")
@api_safe_execute(message="ファンディングレートデータ取得エラー")
async def get_funding_rates(
    symbol: str = Query(
        ..., description="取引ペアシンボル（例: 'BTC/USDT:USDT'）"
    ),
    limit: Optional[int] = Query(
        100, description="取得するデータ数（1-1000）"
    ),
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
        symbol: 取引ペアシンボル（例: 'BTC/USDT:USDT'）
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
    funding_rates = await orchestration_service.get_funding_rate_data(
        symbol=symbol,
        limit=limit or 100,
        start_date=start_date,
        end_date=end_date,
        db_session=db,
    )
    return {
        "success": True,
        "data": {
            "symbol": symbol,
            "count": len(funding_rates),
            "funding_rates": funding_rates,
        },
    }


@router.post("/collect")
@api_safe_execute(message="ファンディングレートデータ収集エラー")
async def collect_funding_rate_data(
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
    orchestration_service: FundingRateOrchestrationService = Depends(
        get_funding_rate_orchestration_service
    ),
    db: Session = Depends(get_db),
):
    """
    ファンディングレートデータを収集してデータベースに保存します

    取引所からファンディングレートデータを取得し、データベースに保存します。

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
    return await orchestration_service.collect_funding_rate_data(
        symbol=symbol,
        limit=limit or 100,
        fetch_all=fetch_all,
        db_session=db,
    )


@router.post("/bulk-collect")
@api_safe_execute(message="ファンディングレート一括収集エラー")
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
    symbols = [
        DEFAULT_MARKET_SYMBOL,
    ]

    return await orchestration_service.collect_bulk_funding_rate_data(
        symbols=symbols, db_session=db
    )
