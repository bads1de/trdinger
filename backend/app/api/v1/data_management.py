"""
データ管理APIルーター

OHLCVデータをデータベースに保存するためのAPIエンドポイントです。
CCXTライブラリを使用してBybit取引所からデータを取得し、データベースに保存します。

@author Trdinger Development Team
@version 1.0.0
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Optional
from datetime import datetime
import logging
from pydantic import BaseModel, Field

from app.core.services.market_data_service import get_market_data_service
from app.config.market_config import MarketDataConfig
import ccxt


# ログ設定
logger = logging.getLogger(__name__)

# ルーター作成
router = APIRouter(prefix="/market-data", tags=["data-management"])


class SaveOHLCVRequest(BaseModel):
    """OHLCVデータ保存リクエストモデル"""
    symbol: str = Field(..., description="取引ペアシンボル（例: BTC/USD:BTC, BTCUSD）")
    timeframe: str = Field(
        MarketDataConfig.DEFAULT_TIMEFRAME,
        description="時間軸（1m, 5m, 15m, 30m, 1h, 4h, 1d）",
    )
    limit: int = Field(
        MarketDataConfig.DEFAULT_LIMIT,
        ge=MarketDataConfig.MIN_LIMIT,
        le=MarketDataConfig.MAX_LIMIT,
        description=f"取得するデータ数（{MarketDataConfig.MIN_LIMIT}-{MarketDataConfig.MAX_LIMIT}）",
    )


@router.post("/save-ohlcv")
async def save_ohlcv_data(request: SaveOHLCVRequest):
    """
    OHLCVデータを取得してデータベースに保存します

    Bybit取引所からOHLCV（Open, High, Low, Close, Volume）データを取得し、
    データベースに保存します。重複データは自動的に無視されます。

    Args:
        request: 保存リクエストデータ

    Returns:
        保存結果を含むJSONレスポンス

    Raises:
        HTTPException: パラメータが無効な場合やAPI呼び出しでエラーが発生した場合
    """
    try:
        logger.info(
            f"OHLCVデータ保存リクエスト: symbol={request.symbol}, "
            f"timeframe={request.timeframe}, limit={request.limit}"
        )

        # 市場データサービスを取得
        service = get_market_data_service()

        # OHLCVデータを取得
        ohlcv_data = await service.fetch_ohlcv_data(
            request.symbol, request.timeframe, request.limit
        )

        # データベースに保存
        records_saved = await service.save_ohlcv_to_database(
            ohlcv_data, request.symbol, request.timeframe
        )

        # シンボルを正規化（レスポンスで使用）
        normalized_symbol = service.normalize_symbol(request.symbol)

        logger.info(f"OHLCVデータ保存完了: {records_saved}件保存")

        # 保存結果に応じたメッセージ
        if records_saved == 0:
            message = f"{normalized_symbol} の {request.timeframe} データは既に存在するため、新規保存はありませんでした"
        else:
            message = f"{normalized_symbol} の {request.timeframe} OHLCVデータを {records_saved}件保存しました"

        return {
            "success": True,
            "records_saved": records_saved,
            "symbol": normalized_symbol,
            "timeframe": request.timeframe,
            "limit": request.limit,
            "message": message,
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
            "message": f"無効なシンボルです: {request.symbol}",
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
