"""
データ収集API

バックテスト用のOHLCVデータ収集エンドポイント
責務の分離により、ビジネスロジックはサービス層に委譲されています。
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict

from app.services.data_collection.orchestration.data_collection_orchestration_service import (
    DataCollectionOrchestrationService,
)
from app.utils.unified_error_handler import UnifiedErrorHandler
from database.connection import get_db, ensure_db_initialized
from app.api.dependencies import get_data_collection_orchestration_service


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data-collection", tags=["data-collection"])


@router.post("/historical")
async def collect_historical_data(
    background_tasks: BackgroundTasks,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    db: Session = Depends(get_db),
    orchestration_service: DataCollectionOrchestrationService = Depends(
        get_data_collection_orchestration_service
    ),
) -> Dict:
    """
    履歴データを包括的に収集
    データベースにデータが存在しない場合のみ新規収集を行います。

    Args:
        symbol: 取引ペア（デフォルト: BTC/USDT）
        timeframe: 時間軸（デフォルト: 1h）
        db: データベースセッション

    Returns:
        収集開始レスポンスまたは既存データ情報
    """

    async def _execute():
        # データベース初期化確認
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました")
            from fastapi import HTTPException

            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        return await orchestration_service.start_historical_data_collection(
            symbol, timeframe, background_tasks, db
        )

    return await UnifiedErrorHandler.safe_execute_async(_execute)


@router.post("/bulk-incremental-update")
async def update_bulk_incremental_data(
    symbol: str = "BTC/USDT:USDT", db: Session = Depends(get_db)
) -> Dict:
    """
    一括差分データを更新（OHLCV、FR、OI）

    OHLCVは全時間足（15m, 30m, 1h, 4h, 1d）を自動的に処理します。

    Args:
        symbol: 取引ペア（デフォルト: BTC/USDT:USDT）
        db: データベースセッション

    Returns:
        一括差分更新結果
    """

    async def _execute():

        orchestration_service = DataCollectionOrchestrationService()
        return await orchestration_service.execute_bulk_incremental_update(symbol, db)

    return await UnifiedErrorHandler.safe_execute_async(_execute)


@router.post("/bitcoin-full")
async def collect_bitcoin_full_data(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    orchestration_service: DataCollectionOrchestrationService = Depends(
        get_data_collection_orchestration_service
    ),
) -> Dict:
    """
    ビットコインの全時間軸データを収集（ベータ版機能）

    Args:
        background_tasks: バックグラウンドタスク
        db: データベースセッション

    Returns:
        収集開始レスポンス
    """

    async def _execute():
        return await orchestration_service.start_bitcoin_full_data_collection(
            background_tasks, db
        )

    return await UnifiedErrorHandler.safe_execute_async(_execute)


@router.post("/bulk-historical")
async def collect_bulk_historical_data(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    orchestration_service: DataCollectionOrchestrationService = Depends(
        get_data_collection_orchestration_service
    ),
) -> Dict:
    """
    全ての取引ペアと全ての時間軸でOHLCVデータを一括収集

    既存データをチェックし、データが存在しない組み合わせのみ収集を実行します。

    Args:
        background_tasks: バックグラウンドタスク
        db: データベースセッション

    Returns:
        一括収集開始レスポンス
    """

    async def _execute():
        # データベース初期化確認
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました")
            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        return await orchestration_service.start_bulk_historical_data_collection(
            background_tasks, db
        )

    return await UnifiedErrorHandler.safe_execute_async(_execute)


@router.get("/status/{symbol:path}/{timeframe}")
async def get_collection_status(
    symbol: str,
    timeframe: str,
    background_tasks: BackgroundTasks,
    auto_fetch: bool = False,
    db: Session = Depends(get_db),
) -> Dict:
    """
    データ収集状況を確認

    Args:
        symbol: 取引ペア
        timeframe: 時間軸
        auto_fetch: データが存在しない場合に自動フェッチを開始するか
        background_tasks: バックグラウンドタスク
        db: データベースセッション

    Returns:
        データ収集状況
    """

    async def _get_collection_status():
        # データベース初期化確認
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました", exc_info=True)
            from fastapi import HTTPException

            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        service = DataCollectionOrchestrationService()
        return await service.get_collection_status(
            symbol=symbol,
            timeframe=timeframe,
            background_tasks=background_tasks,
            auto_fetch=auto_fetch,
            db=db,
        )

    return await UnifiedErrorHandler.safe_execute_async(_get_collection_status)


@router.post("/all/bulk-collect")
async def collect_all_data_bulk(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    orchestration_service: DataCollectionOrchestrationService = Depends(
        get_data_collection_orchestration_service
    ),
) -> Dict:
    """
    全データ（OHLCV・Funding Rate・Open Interest）を一括収集

    既存データをチェックし、データが存在しない組み合わせのみ収集を実行します。

    Args:
        background_tasks: バックグラウンドタスク
        db: データベースセッション

    Returns:
        全データ一括収集開始レスポンス
    """

    async def _execute():
        # データベース初期化確認
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました")
            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        return await orchestration_service.start_all_data_bulk_collection(
            background_tasks, db
        )

    return await UnifiedErrorHandler.safe_execute_async(_execute)
