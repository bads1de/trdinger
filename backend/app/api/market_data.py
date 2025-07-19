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
from database.repositories.ohlcv_repository import OHLCVRepository
from app.core.utils.api_utils import DateTimeHelper, APIResponseHelper
from app.core.utils.unified_error_handler import UnifiedErrorHandler
from app.core.utils.data_converter import OHLCVDataConverter


# ログ設定
logger = logging.getLogger(__name__)

# ルーター作成
router = APIRouter(prefix="/market-data", tags=["market-data"])


@router.get("/ohlcv")
async def get_ohlcv_data(
    symbol: str = Query(..., description="取引ペアシンボル（例: BTC/USDT）"),
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
        logger.info(
            f"OHLCVデータ取得リクエスト: symbol={symbol}, timeframe={timeframe}, limit={limit}"
        )

        repository = OHLCVRepository(db)

        start_time = None
        end_time = None

        if start_date:
            start_time = DateTimeHelper.parse_iso_datetime(start_date)
        if end_date:
            end_time = DateTimeHelper.parse_iso_datetime(end_date)

        if start_time is None and end_time is None:
            ohlcv_records = repository.get_latest_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
            )
        else:
            ohlcv_records = repository.get_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )

        ohlcv_data = OHLCVDataConverter.db_to_api_format(ohlcv_records)

        logger.info(f"OHLCVデータ取得成功: {len(ohlcv_data)}件")

        return APIResponseHelper.api_response(
            success=True,
            data={"ohlcv_data": ohlcv_data, "symbol": symbol, "timeframe": timeframe},
            message=f"{symbol} の {timeframe} OHLCVデータを取得しました",
        )

    return await UnifiedErrorHandler.safe_execute_async(
        _get_ohlcv, message="OHLCVデータ取得エラー"
    )
