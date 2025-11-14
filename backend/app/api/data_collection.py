"""
データ収集API

バックテスト用のOHLCVデータ収集エンドポイント
責務の分離により、ビジネスロジックはサービス層に委譲されています。
"""

import logging
from typing import Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_data_collection_orchestration_service
from app.services.data_collection.orchestration.data_collection_orchestration_service import (
    DataCollectionOrchestrationService,
)
from app.utils.error_handler import ErrorHandler
from database.connection import get_db, init_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data-collection", tags=["data-collection"])


@router.post("/historical")
async def collect_historical_data(
    background_tasks: BackgroundTasks,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    force_update: bool = False,
    start_date: Optional[str] = None,
    db: Session = Depends(get_db),
    orchestration_service: DataCollectionOrchestrationService = Depends(
        get_data_collection_orchestration_service
    ),
) -> Dict:
    """
    履歴データを包括的に収集

    Args:
        symbol: 取引ペア（デフォルト: BTC/USDT）
        timeframe: 時間軸（デフォルト: 1h）
        force_update: 強制更新（データが存在しても上書き）
        start_date: 開始日付（YYYY-MM-DD形式、指定しない場合は2020-03-25）
        db: データベースセッション

    Returns:
        収集開始レスポンスまたは既存データ情報
    """

    async def _execute():
        # データベース初期化確認
        if not init_db():
            logger.error("データベースの初期化に失敗しました")
            from fastapi import HTTPException

            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        return await orchestration_service.start_historical_data_collection(
            symbol, timeframe, background_tasks, db, force_update, start_date
        )

    return await ErrorHandler.safe_execute_async(_execute)


@router.post("/bulk-incremental-update")
async def update_bulk_incremental_data(
    symbol: str = "BTC/USDT:USDT",
    db: Session = Depends(get_db),
    orchestration_service: DataCollectionOrchestrationService = Depends(
        get_data_collection_orchestration_service
    ),
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

        return await orchestration_service.execute_bulk_incremental_update(symbol, db)

    return await ErrorHandler.safe_execute_async(_execute)


@router.post("/bulk-historical")
async def collect_bulk_historical_data(
    background_tasks: BackgroundTasks,
    force_update: bool = True,
    start_date: str = "2020-03-25",
    db: Session = Depends(get_db),
    orchestration_service: DataCollectionOrchestrationService = Depends(
        get_data_collection_orchestration_service
    ),
) -> Dict:
    """
    全ての取引ペアと全ての時間軸でOHLCVデータを一括収集

    Args:
        background_tasks: バックグラウンドタスク
        force_update: 強制更新（データが存在しても上書き）
        start_date: 開始日付（YYYY-MM-DD形式、指定しない場合は2020-03-25）
        db: データベースセッション

    Returns:
        一括収集開始レスポンス
    """

    async def _execute():
        # データベース初期化確認
        if not init_db():
            logger.error("データベースの初期化に失敗しました")
            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        return await orchestration_service.start_bulk_historical_data_collection(
            background_tasks, db, force_update, start_date
        )

    return await ErrorHandler.safe_execute_async(_execute)


@router.get("/status/{symbol:path}/{timeframe}")
async def get_collection_status(
    symbol: str,
    timeframe: str,
    background_tasks: BackgroundTasks,
    auto_fetch: bool = False,
    db: Session = Depends(get_db),
    orchestration_service: DataCollectionOrchestrationService = Depends(
        get_data_collection_orchestration_service
    ),
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
        if not init_db():
            logger.error("データベースの初期化に失敗しました", exc_info=True)
            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        return await orchestration_service.get_collection_status(
            symbol=symbol,
            timeframe=timeframe,
            background_tasks=background_tasks,
            auto_fetch=auto_fetch,
            db=db,
        )

    return await ErrorHandler.safe_execute_async(_get_collection_status)


@router.post("/all/bulk-collect")
async def collect_all_data_bulk(
    background_tasks: BackgroundTasks,
    force_update: bool = False,
    start_date: Optional[str] = None,
    db: Session = Depends(get_db),
    orchestration_service: DataCollectionOrchestrationService = Depends(
        get_data_collection_orchestration_service
    ),
) -> Dict:
    """
    全データ（OHLCV・Funding Rate・Open Interest）を一括収集

    Args:
        background_tasks: バックグラウンドタスク
        force_update: 強制更新（データが存在しても上書き）
        start_date: 開始日付（YYYY-MM-DD形式、指定しない場合は2020-03-25）
        db: データベースセッション

    Returns:
        全データ一括収集開始レスポンス
    """

    async def _execute():
        # データベース初期化確認
        if not init_db():
            logger.error("データベースの初期化に失敗しました")
            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        # 全データ一括収集サービスにも上書きオプションを追加する必要があるため、
        # 一旦はbulk-historicalを使用する
        return await orchestration_service.start_bulk_historical_data_collection(
            background_tasks, db, force_update, start_date
        )

    return await ErrorHandler.safe_execute_async(_execute)
