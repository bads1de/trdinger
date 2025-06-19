"""
データリセットAPI

OHLCV、ファンディングレート、オープンインタレストデータのリセット機能を提供します。
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any
from datetime import datetime
import logging

from database.connection import get_db
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from app.core.utils.api_utils import APIResponseHelper, APIErrorHandler

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

    return await APIErrorHandler.handle_api_exception(
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

    return await APIErrorHandler.handle_api_exception(
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

    return await APIErrorHandler.handle_api_exception(
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

    return await APIErrorHandler.handle_api_exception(
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

    return await APIErrorHandler.handle_api_exception(
        _reset_by_symbol, message="シンボル別データのリセット中にエラーが発生しました"
    )


@router.get("/status")
async def get_data_status(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    現在のデータ状況を取得

    Returns:
        各データタイプの件数情報
    """

    async def _get_status():
        # リポジトリインスタンス作成
        OHLCVRepository(db)
        FundingRateRepository(db)
        OpenInterestRepository(db)

        # 各データの件数取得
        from database.models import OHLCVData, FundingRateData, OpenInterestData

        ohlcv_count = db.query(OHLCVData).count()
        fr_count = db.query(FundingRateData).count()
        oi_count = db.query(OpenInterestData).count()

        response_data = {
            "data_counts": {
                "ohlcv": ohlcv_count,
                "funding_rates": fr_count,
                "open_interest": oi_count,
            },
            "total_records": ohlcv_count + fr_count + oi_count,
        }

        return APIResponseHelper.api_response(
            success=True,
            data=response_data,
            message="現在のデータ状況を取得しました",
        )

    return await APIErrorHandler.handle_api_exception(
        _get_status, message="データステータスの取得中にエラーが発生しました"
    )
