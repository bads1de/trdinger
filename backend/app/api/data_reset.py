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
        # リポジトリインスタンス作成
        ohlcv_repo = OHLCVRepository(db)
        fr_repo = FundingRateRepository(db)
        oi_repo = OpenInterestRepository(db)

        # 各データの削除実行
        deleted_counts = {}
        errors = []

        # OHLCVデータ削除
        try:
            deleted_counts["ohlcv"] = ohlcv_repo.clear_all_ohlcv_data()
        except Exception as e:
            errors.append(f"OHLCV削除エラー: {str(e)}")
            deleted_counts["ohlcv"] = 0

        # ファンディングレートデータ削除
        try:
            deleted_counts["funding_rates"] = fr_repo.clear_all_funding_rate_data()
        except Exception as e:
            errors.append(f"ファンディングレート削除エラー: {str(e)}")
            deleted_counts["funding_rates"] = 0

        # オープンインタレストデータ削除
        try:
            deleted_counts["open_interest"] = oi_repo.clear_all_open_interest_data()
        except Exception as e:
            errors.append(f"オープンインタレスト削除エラー: {str(e)}")
            deleted_counts["open_interest"] = 0

        # 結果の集計
        total_deleted = sum(deleted_counts.values())
        success = len(errors) == 0

        response = {
            "success": success,
            "deleted_counts": deleted_counts,
            "total_deleted": total_deleted,
            "message": (
                "全データのリセットが完了しました"
                if success
                else "一部のデータリセットでエラーが発生しました"
            ),
            "errors": errors,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"全データリセット完了: {deleted_counts}")
        return response

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
        ohlcv_repo = OHLCVRepository(db)
        deleted_count = ohlcv_repo.clear_all_ohlcv_data()

        response = {
            "success": True,
            "deleted_count": deleted_count,
            "data_type": "ohlcv",
            "message": f"OHLCVデータを{deleted_count}件削除しました",
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"OHLCVデータリセット完了: {deleted_count}件")
        return response

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
        fr_repo = FundingRateRepository(db)
        deleted_count = fr_repo.clear_all_funding_rate_data()

        response = {
            "success": True,
            "deleted_count": deleted_count,
            "data_type": "funding_rates",
            "message": f"ファンディングレートデータを{deleted_count}件削除しました",
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"ファンディングレートデータリセット完了: {deleted_count}件")
        return response

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
        oi_repo = OpenInterestRepository(db)
        deleted_count = oi_repo.clear_all_open_interest_data()

        message = f"オープンインタレストデータを{deleted_count}件削除しました"

        logger.info(f"オープンインタレストデータリセット完了: {deleted_count}件")
        return APIResponseHelper.api_response(
            success=True,
            data={
                "deleted_count": deleted_count,
                "data_type": "open_interest",
            },
            message=message,
        )

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
        ohlcv_repo = OHLCVRepository(db)
        fr_repo = FundingRateRepository(db)
        oi_repo = OpenInterestRepository(db)

        deleted_counts = {}
        errors = []

        try:
            deleted_counts["ohlcv"] = ohlcv_repo.clear_ohlcv_data_by_symbol(symbol)
        except Exception as e:
            errors.append(f"OHLCV削除エラー: {str(e)}")
            deleted_counts["ohlcv"] = 0

        try:
            deleted_counts["funding_rates"] = fr_repo.clear_funding_rate_data_by_symbol(
                symbol
            )
        except Exception as e:
            errors.append(f"ファンディングレート削除エラー: {str(e)}")
            deleted_counts["funding_rates"] = 0

        try:
            deleted_counts["open_interest"] = (
                oi_repo.clear_open_interest_data_by_symbol(symbol)
            )
        except Exception as e:
            errors.append(f"オープンインタレスト削除エラー: {str(e)}")
            deleted_counts["open_interest"] = 0

        total_deleted = sum(deleted_counts.values())
        success = len(errors) == 0

        message = (
            f"シンボル '{symbol}' のデータリセットが完了しました"
            if success
            else f"シンボル '{symbol}' の一部データリセットでエラーが発生しました"
        )

        logger.info(f"シンボル '{symbol}' データリセット完了: {deleted_counts}")
        return APIResponseHelper.api_response(
            success=success,
            data={
                "symbol": symbol,
                "deleted_counts": deleted_counts,
                "total_deleted": total_deleted,
                "errors": errors,
            },
            message=message,
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
