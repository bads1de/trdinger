"""
データリセットAPI

OHLCV、ファンディングレート、オープンインタレストデータのリセット機能を提供します。
"""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any
from datetime import datetime


from database.connection import get_db
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from app.core.utils.api_utils import APIResponseHelper
from app.core.utils.unified_error_handler import UnifiedErrorHandler
from app.core.services.data_collection.data_management_orchestration_service import (
    DataManagementOrchestrationService,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data-reset", tags=["data-reset"])


@router.delete("/all")
async def reset_all_data(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    全てのデータ（OHLCV、ファンディングレート、オープンインタレスト）をリセット

    Returns:
        削除結果の詳細
    """

    async def _reset_all_data():
        orchestration_service = DataManagementOrchestrationService()
        return await orchestration_service.reset_all_data(db_session=db)

    return await UnifiedErrorHandler.safe_execute_async(
        _reset_all_data, message="全データのリセット中にエラーが発生しました"
    )


@router.delete("/ohlcv")
async def reset_ohlcv_data(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    OHLCVデータのみをリセット

    Returns:
        削除結果の詳細
    """

    async def _reset_ohlcv_data():
        orchestration_service = DataManagementOrchestrationService()
        return await orchestration_service.reset_ohlcv_data(db_session=db)

    return await UnifiedErrorHandler.safe_execute_async(
        _reset_ohlcv_data, message="OHLCVデータのリセット中にエラーが発生しました"
    )


@router.delete("/funding-rates")
async def reset_funding_rate_data(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    ファンディングレートデータのみをリセット

    Returns:
        削除結果の詳細
    """

    async def _reset_funding_rate_data():
        orchestration_service = DataManagementOrchestrationService()
        return await orchestration_service.reset_funding_rate_data(db_session=db)

    return await UnifiedErrorHandler.safe_execute_async(
        _reset_funding_rate_data,
        message="ファンディングレートデータのリセット中にエラーが発生しました",
    )


@router.delete("/open-interest")
async def reset_open_interest_data(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    オープンインタレストデータのみをリセット

    Returns:
        削除結果の詳細
    """

    async def _reset_open_interest():
        orchestration_service = DataManagementOrchestrationService()
        return await orchestration_service.reset_open_interest_data(db_session=db)

    return await UnifiedErrorHandler.safe_execute_async(
        _reset_open_interest,
        message="オープンインタレストデータのリセット中にエラーが発生しました",
    )


@router.delete("/symbol/{symbol}")
async def reset_data_by_symbol(
    symbol: str, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    特定シンボルの全データ（OHLCV、ファンディングレート、オープンインタレスト）をリセット

    Args:
        symbol: 削除対象のシンボル（例: BTC/USDT:USDT）

    Returns:
        削除結果の詳細
    """

    async def _reset_by_symbol():
        orchestration_service = DataManagementOrchestrationService()
        return await orchestration_service.reset_data_by_symbol(
            symbol=symbol, db_session=db
        )

    return await UnifiedErrorHandler.safe_execute_async(
        _reset_by_symbol, message="シンボル別データのリセット中にエラーが発生しました"
    )


@router.get("/status")
async def get_data_status(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    現在のデータ状況を取得（詳細版）

    Returns:
        各データタイプの詳細情報（件数、最新・最古データ）
    """

    async def _get_status():
        # リポジトリインスタンス作成
        ohlcv_repo = OHLCVRepository(db)
        fr_repo = FundingRateRepository(db)
        oi_repo = OpenInterestRepository(db)

        # 各データの件数取得
        from database.models import (
            OHLCVData,
            FundingRateData,
            OpenInterestData,
            FearGreedIndexData,
            ExternalMarketData,
        )

        ohlcv_count = db.query(OHLCVData).count()
        fr_count = db.query(FundingRateData).count()
        oi_count = db.query(OpenInterestData).count()
        fg_count = db.query(FearGreedIndexData).count()
        em_count = db.query(ExternalMarketData).count()

        # OHLCV詳細情報（時間足別）
        timeframes = ["15m", "30m", "1h", "4h", "1d"]
        symbol = "BTC/USDT:USDT"

        ohlcv_details = {}
        for tf in timeframes:
            count = ohlcv_repo.get_data_count(symbol, tf)
            latest = ohlcv_repo.get_latest_timestamp(symbol, tf)
            oldest = ohlcv_repo.get_oldest_timestamp(symbol, tf)

            ohlcv_details[tf] = {
                "count": count,
                "latest_timestamp": latest.isoformat() if latest else None,
                "oldest_timestamp": oldest.isoformat() if oldest else None,
            }

        # ファンディングレート詳細情報
        fr_latest = fr_repo.get_latest_funding_timestamp(symbol)
        fr_oldest = fr_repo.get_oldest_funding_timestamp(symbol)

        # オープンインタレスト詳細情報
        oi_latest = oi_repo.get_latest_open_interest_timestamp(symbol)
        oi_oldest = oi_repo.get_oldest_open_interest_timestamp(symbol)

        # 外部市場データ詳細情報
        from database.repositories.external_market_repository import (
            ExternalMarketRepository,
        )

        em_repo = ExternalMarketRepository(db)
        em_latest = em_repo.get_latest_data_timestamp()
        em_statistics = em_repo.get_data_statistics()

        response_data = {
            "data_counts": {
                "ohlcv": ohlcv_count,
                "funding_rates": fr_count,
                "open_interest": oi_count,
                "fear_greed_index": fg_count,
                "external_market_data": em_count,
            },
            "total_records": ohlcv_count + fr_count + oi_count + fg_count + em_count,
            "details": {
                "ohlcv": {
                    "symbol": symbol,
                    "timeframes": ohlcv_details,
                    "total_count": ohlcv_count,
                },
                "funding_rates": {
                    "symbol": symbol,
                    "count": fr_count,
                    "latest_timestamp": fr_latest.isoformat() if fr_latest else None,
                    "oldest_timestamp": fr_oldest.isoformat() if fr_oldest else None,
                },
                "open_interest": {
                    "symbol": symbol,
                    "count": oi_count,
                    "latest_timestamp": oi_latest.isoformat() if oi_latest else None,
                    "oldest_timestamp": oi_oldest.isoformat() if oi_oldest else None,
                },
                "external_market_data": {
                    "count": em_count,
                    "symbols": em_statistics.get("symbols", []),
                    "symbol_count": em_statistics.get("symbol_count", 0),
                    "latest_timestamp": em_latest.isoformat() if em_latest else None,
                    "date_range": em_statistics.get("date_range"),
                },
            },
        }

        return APIResponseHelper.api_response(
            success=True,
            data=response_data,
            message="現在のデータ状況を取得しました",
        )

    return await UnifiedErrorHandler.safe_execute_async(
        _get_status, message="データステータスの取得中にエラーが発生しました"
    )
