"""
市場データAPIルーター

データベースからのOHLCVデータ取得APIエンドポイントです。
バックテスト用に保存されたデータを提供し、適切なエラーハンドリングとバリデーションを含みます。

"""

import logging

from fastapi import APIRouter, Query, Depends
from typing import Optional
from sqlalchemy.orm import Session

from app.config.unified_config import unified_config
from database.connection import get_db
from app.core.services.data_collection.orchestration.market_data_orchestration_service import (
    MarketDataOrchestrationService,
)
from app.core.utils.unified_error_handler import UnifiedErrorHandler


# ログ設定
logger = logging.getLogger(__name__)

# ルーター作成
router = APIRouter(prefix="/api/market-data", tags=["market-data"])


@router.get("/ohlcv")
async def get_ohlcv_data(
    symbol: str = Query(..., description="取引ペアシンボル（例: BTC/USDT:USDT）"),
    timeframe: str = Query(
        unified_config.market.default_timeframe,
        description="時間軸（1m, 5m, 15m, 30m, 1h, 4h, 1d）",
    ),
    limit: int = Query(
        unified_config.market.default_limit,
        ge=unified_config.market.min_limit,
        le=unified_config.market.max_limit,
        description=f"取得するデータ数（{unified_config.market.min_limit}-{unified_config.market.max_limit}）",
    ),
    start_date: Optional[str] = Query(None, description="開始日時（ISO形式）"),
    end_date: Optional[str] = Query(None, description="終了日時（ISO形式）"),
    db: Session = Depends(get_db),
):
    """
    OHLCVデータを取得します

    データベースに保存されたOHLCV（Open, High, Low, Close, Volume）データを取得します。

    Args:
        symbol: 取引ペアシンボル（例: 'BTC/USDT'）
        timeframe: 時間軸（例: '1h', '1d'）
        limit: 取得するデータ数（1-1000）
        start_date: 開始日時（ISO形式）
        end_date: 終了日時（ISO形式）
        db: データベースセッション

    Returns:
        OHLCVデータを含むJSONレスポンス

    Raises:
        HTTPException: パラメータが無効な場合やデータベースエラーが発生した場合
    """

    async def _get_ohlcv():
        service = MarketDataOrchestrationService(db)
        return await service.get_ohlcv_data(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
        )

    return await UnifiedErrorHandler.safe_execute_async(
        _get_ohlcv, message="OHLCVデータ取得エラー"
    )
