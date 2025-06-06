"""
市場データAPIルーター

データベースからのOHLCVデータ取得APIエンドポイントです。
バックテスト用に保存されたデータを提供し、適切なエラーハンドリングとバリデーションを含みます。

@author Trdinger Development Team
@version 2.0.0
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
import logging

from app.config.market_config import MarketDataConfig
from database.connection import get_db
from database.repositories.ohlcv_repository import OHLCVRepository
from app.core.utils.api_utils import APIResponseHelper, APIErrorHandler, DateTimeHelper
from app.core.utils.data_converter import OHLCVDataConverter


# ログ設定
logger = logging.getLogger(__name__)

# ルーター作成
router = APIRouter(prefix="/market-data", tags=["market-data"])


@router.get("/ohlcv")
async def get_ohlcv_data(
    symbol: str = Query(..., description="取引ペアシンボル（例: BTC/USDT）"),
    timeframe: str = Query(
        MarketDataConfig.DEFAULT_TIMEFRAME,
        description="時間軸（1m, 5m, 15m, 30m, 1h, 4h, 1d）",
    ),
    limit: int = Query(
        MarketDataConfig.DEFAULT_LIMIT,
        ge=MarketDataConfig.MIN_LIMIT,
        le=MarketDataConfig.MAX_LIMIT,
        description=f"取得するデータ数（{MarketDataConfig.MIN_LIMIT}-{MarketDataConfig.MAX_LIMIT}）",
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
    try:
        logger.info(
            f"OHLCVデータ取得リクエスト: symbol={symbol}, timeframe={timeframe}, limit={limit}"
        )

        # データベースリポジトリを作成
        repository = OHLCVRepository(db)

        # 日付パラメータの変換
        start_time = None
        end_time = None

        if start_date:
            start_time = DateTimeHelper.parse_iso_datetime(start_date)
        if end_date:
            end_time = DateTimeHelper.parse_iso_datetime(end_date)

        # データベースからOHLCVデータを取得
        ohlcv_records = repository.get_ohlcv_data(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

        # データをAPIレスポンス形式に変換
        ohlcv_data = OHLCVDataConverter.db_to_api_format(ohlcv_records)

        logger.info(f"OHLCVデータ取得成功: {len(ohlcv_data)}件")

        return APIResponseHelper.success_response(
            data=ohlcv_data,
            message=f"{symbol} の {timeframe} OHLCVデータを取得しました",
            additional_fields={
                "symbol": symbol,
                "timeframe": timeframe
            }
        )

    except ValueError as e:
        raise APIErrorHandler.handle_validation_error(e, "OHLCVデータ取得")

    except Exception as e:
        raise APIErrorHandler.handle_database_error(e, "OHLCVデータ取得")


@router.get("/symbols")
async def get_supported_symbols():
    """
    サポートされている取引ペアシンボルの一覧を取得します

    Returns:
        サポートされているシンボルのリスト
    """
    try:
        return APIResponseHelper.success_response(
            data={
                "symbols": MarketDataConfig.SUPPORTED_SYMBOLS,
                "symbol_mapping": MarketDataConfig.SYMBOL_MAPPING,
            },
            message="サポートされているシンボル一覧を取得しました"
        )
    except Exception as e:
        raise APIErrorHandler.handle_generic_error(e, "シンボル一覧取得")


@router.get("/timeframes")
async def get_supported_timeframes():
    """
    サポートされている時間軸の一覧を取得します

    Returns:
        サポートされている時間軸のリスト
    """
    try:
        return APIResponseHelper.success_response(
            data={
                "timeframes": MarketDataConfig.SUPPORTED_TIMEFRAMES,
                "default": MarketDataConfig.DEFAULT_TIMEFRAME,
            },
            message="サポートされている時間軸一覧を取得しました"
        )
    except Exception as e:
        raise APIErrorHandler.handle_generic_error(e, "時間軸一覧取得")
