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
from database.repository import OHLCVRepository


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
            start_time = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        if end_date:
            end_time = datetime.fromisoformat(end_date.replace("Z", "+00:00"))

        # データベースからOHLCVデータを取得
        ohlcv_records = repository.get_ohlcv_data(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

        # データをAPIレスポンス形式に変換
        ohlcv_data = []
        for record in ohlcv_records:
            ohlcv_data.append(
                [
                    int(
                        record.timestamp.timestamp() * 1000
                    ),  # タイムスタンプ（ミリ秒）
                    record.open,
                    record.high,
                    record.low,
                    record.close,
                    record.volume,
                ]
            )

        logger.info(f"OHLCVデータ取得成功: {len(ohlcv_data)}件")

        return {
            "success": True,
            "data": ohlcv_data,
            "symbol": symbol,
            "timeframe": timeframe,
            "message": f"{symbol} の {timeframe} OHLCVデータを取得しました",
            "timestamp": datetime.now().isoformat(),
        }

    except ValueError as e:
        logger.error(f"バリデーションエラー: {e}")
        error_response = {
            "success": False,
            "message": str(e),
            "timestamp": datetime.now().isoformat(),
        }
        raise HTTPException(status_code=400, detail=error_response)

    except Exception as e:
        logger.error(f"データベースエラー: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "データベースエラーが発生しました。",
                "timestamp": datetime.now().isoformat(),
            },
        )


@router.get("/symbols")
async def get_supported_symbols():
    """
    サポートされている取引ペアシンボルの一覧を取得します

    Returns:
        サポートされているシンボルのリスト
    """
    try:
        return {
            "success": True,
            "data": {
                "symbols": MarketDataConfig.SUPPORTED_SYMBOLS,
                "symbol_mapping": MarketDataConfig.SYMBOL_MAPPING,
            },
            "message": "サポートされているシンボル一覧を取得しました",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        logger.error(f"シンボル一覧取得エラー: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "シンボル一覧の取得でエラーが発生しました。",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )


@router.get("/timeframes")
async def get_supported_timeframes():
    """
    サポートされている時間軸の一覧を取得します

    Returns:
        サポートされている時間軸のリスト
    """
    try:
        return {
            "success": True,
            "data": {
                "timeframes": MarketDataConfig.SUPPORTED_TIMEFRAMES,
                "default": MarketDataConfig.DEFAULT_TIMEFRAME,
            },
            "message": "サポートされている時間軸一覧を取得しました",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        logger.error(f"時間軸一覧取得エラー: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "時間軸一覧の取得でエラーが発生しました。",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )
