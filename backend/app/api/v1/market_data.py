"""
市場データAPIルーター

CCXT ライブラリを使用したBybit取引所からのOHLCVデータ取得APIエンドポイントです。
リアルタイムの市場データを提供し、適切なエラーハンドリングとバリデーションを含みます。

@author Trdinger Development Team
@version 1.0.0
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
import logging

from app.core.services.market_data_service import get_market_data_service
from app.config.market_config import MarketDataConfig
import ccxt


# ログ設定
logger = logging.getLogger(__name__)

# ルーター作成
router = APIRouter(prefix="/market-data", tags=["market-data"])


@router.get("/ohlcv")
async def get_ohlcv_data(
    symbol: str = Query(..., description="取引ペアシンボル（例: BTC/USD:BTC, BTCUSD）"),
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
):
    """
    OHLCVデータを取得します

    Bybit取引所からリアルタイムのOHLCV（Open, High, Low, Close, Volume）データを取得します。

    Args:
        symbol: 取引ペアシンボル（例: 'BTC/USD:BTC', 'BTCUSD'）
        timeframe: 時間軸（例: '1h', '1d'）
        limit: 取得するデータ数（1-1000）

    Returns:
        OHLCVデータを含むJSONレスポンス

    Raises:
        HTTPException: パラメータが無効な場合やAPI呼び出しでエラーが発生した場合
    """
    try:
        logger.info(
            f"OHLCVデータ取得リクエスト: symbol={symbol}, timeframe={timeframe}, limit={limit}"
        )

        # 市場データサービスを取得
        service = get_market_data_service()

        # OHLCVデータを取得
        ohlcv_data = await service.fetch_ohlcv_data(symbol, timeframe, limit)

        # シンボルを正規化（レスポンスで使用）
        normalized_symbol = service.normalize_symbol(symbol)

        logger.info(f"OHLCVデータ取得成功: {len(ohlcv_data)}件")

        return {
            "success": True,
            "data": ohlcv_data,
            "symbol": normalized_symbol,
            "timeframe": timeframe,
            "message": f"{normalized_symbol} の {timeframe} OHLCVデータを取得しました",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    except ValueError as e:
        logger.error(f"バリデーションエラー: {e}")
        error_response = {
            "success": False,
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        raise HTTPException(status_code=400, detail=error_response)

    except ccxt.BadSymbol as e:
        logger.error(f"無効なシンボル: {e}")
        error_response = {
            "success": False,
            "message": f"無効なシンボルです: {symbol}",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        raise HTTPException(status_code=400, detail=error_response)

    except ccxt.NetworkError as e:
        logger.error(f"ネットワークエラー: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "message": "取引所への接続でエラーが発生しました。しばらく後に再試行してください。",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )

    except ccxt.RateLimitExceeded as e:
        logger.error(f"レート制限エラー: {e}")
        raise HTTPException(
            status_code=429,
            detail={
                "success": False,
                "message": "リクエスト制限に達しました。しばらく後に再試行してください。",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )

    except ccxt.ExchangeError as e:
        logger.error(f"取引所エラー: {e}")
        raise HTTPException(
            status_code=502,
            detail={
                "success": False,
                "message": "取引所でエラーが発生しました。",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )

    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "内部サーバーエラーが発生しました。",
                "timestamp": datetime.utcnow().isoformat() + "Z",
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
