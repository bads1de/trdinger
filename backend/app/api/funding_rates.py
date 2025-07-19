"""
ファンディングレートAPI

ファンディングレートデータの取得・収集機能を提供するAPIエンドポイント
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional


from database.connection import get_db, ensure_db_initialized
from database.repositories.funding_rate_repository import FundingRateRepository
from app.core.services.data_collection.bybit.funding_rate_service import (
    BybitFundingRateService,
)
from app.core.utils.api_utils import APIResponseHelper
from app.core.utils.unified_error_handler import UnifiedErrorHandler
from app.core.services.data_collection.orchestration.funding_rate_orchestration_service import (
    FundingRateOrchestrationService,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["funding-rates"])


@router.get("/funding-rates")
async def get_funding_rates(
    symbol: str = Query(..., description="取引ペアシンボル（例: 'BTC/USDT'）"),
    limit: Optional[int] = Query(100, description="取得するデータ数（1-1000）"),
    start_date: Optional[str] = Query(None, description="開始日時（ISO形式）"),
    end_date: Optional[str] = Query(None, description="終了日時（ISO形式）"),
    db: Session = Depends(get_db),
):
    """
    ファンディングレートデータを取得します

    データベースに保存されたファンディングレートデータを取得します。

    Args:
        symbol: 取引ペアシンボル（例: 'BTC/USDT'）
        limit: 取得するデータ数（1-1000）
        start_date: 開始日時（ISO形式）
        end_date: 終了日時（ISO形式）
        db: データベースセッション

    Returns:
        ファンディングレートデータを含むJSONレスポンス

    Raises:
        HTTPException: パラメータが無効な場合やデータベースエラーが発生した場合
    """

    async def _get_funding_rates_data():
        orchestration_service = FundingRateOrchestrationService()
        return await orchestration_service.get_funding_rate_data(
            symbol=symbol,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            db_session=db,
        )

    return await UnifiedErrorHandler.safe_execute_async(
        _get_funding_rates_data, message="ファンディングレートデータ取得エラー"
    )


@router.post("/funding-rates/collect")
async def collect_funding_rate_data(
    symbol: str = Query(..., description="取引ペアシンボル（例: 'BTC/USDT'）"),
    limit: Optional[int] = Query(
        100, description="取得するデータ数（1-1000、fetch_all=trueの場合は無視）"
    ),
    fetch_all: bool = Query(False, description="全期間のデータを取得するかどうか"),
    db: Session = Depends(get_db),
):
    """
    ファンディングレートデータを収集してデータベースに保存します

    Bybit取引所からファンディングレートデータを取得し、データベースに保存します。

    Args:
        symbol: 取引ペアシンボル（例: 'BTC/USDT'）
        limit: 取得するデータ数（1-1000、fetch_all=trueの場合は無視）
        fetch_all: 全期間のデータを取得するかどうか
        db: データベースセッション

    Returns:
        収集結果を含むJSONレスポンス

    Raises:
        HTTPException: パラメータが無効な場合やAPI/データベースエラーが発生した場合
    """

    async def _collect_rates():
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました")
            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        orchestration_service = FundingRateOrchestrationService()
        return await orchestration_service.collect_funding_rate_data(
            symbol=symbol,
            limit=limit,
            fetch_all=fetch_all,
            db_session=db,
        )

    return await UnifiedErrorHandler.safe_execute_async(
        _collect_rates, message="ファンディングレートデータ収集エラー"
    )


@router.post("/funding-rates/bulk-collect")
async def bulk_collect_funding_rates(
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

    async def _bulk_collect():
        logger.info("ファンディングレート一括収集開始: BTC全期間データ")

        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました")
            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        symbols = [
            "BTC/USDT:USDT",
        ]

        service = BybitFundingRateService()
        repository = FundingRateRepository(db)

        results = []
        total_saved = 0
        successful_symbols = 0
        failed_symbols = []

        for symbol in symbols:
            try:
                result = await service.fetch_and_save_funding_rate_data(
                    symbol=symbol,
                    repository=repository,
                    fetch_all=True,
                )
                results.append(result)
                total_saved += result["saved_count"]
                successful_symbols += 1

                logger.info(f"✅ {symbol}: {result['saved_count']}件保存")

                import asyncio

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"❌ {symbol} 収集エラー: {e}")
                failed_symbols.append({"symbol": symbol, "error": str(e)})

        logger.info(
            f"ファンディングレート一括収集完了: {successful_symbols}/{len(symbols)}成功"
        )

        return APIResponseHelper.api_response(
            data={
                "total_symbols": len(symbols),
                "successful_symbols": successful_symbols,
                "failed_symbols": len(failed_symbols),
                "total_saved_records": total_saved,
                "results": results,
                "failures": failed_symbols,
            },
            success=True,
            message=f"{successful_symbols}/{len(symbols)}シンボル（BTC）で合計{total_saved}件のファンディングレートデータを保存しました",
        )

    return await UnifiedErrorHandler.safe_execute_async(
        _bulk_collect, message="ファンディングレート一括収集エラー"
    )
