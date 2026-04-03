"""
データ収集API

バックテスト用のOHLCVデータ収集エンドポイント
責務の分離により、ビジネスロジックはサービス層に委譲されています。
"""

from typing import Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_data_collection_orchestration_service
from app.config.constants import DEFAULT_MARKET_SYMBOL
from app.services.data_collection.orchestration.data_collection_orchestration_service import (
    DataCollectionOrchestrationService,
)
from app.utils.error_handler import api_safe_execute, ensure_db_initialized
from database.connection import get_db

router = APIRouter(
    prefix="/api/data-collection",
    tags=["data-collection"],
    dependencies=[Depends(ensure_db_initialized)],
)


@router.post("/historical")
@api_safe_execute(message="履歴データ収集操作エラー")
async def collect_historical_data(
    background_tasks: BackgroundTasks,
    symbol: str = DEFAULT_MARKET_SYMBOL,
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
    """
    return await orchestration_service.start_historical_data_collection(
        symbol, timeframe, background_tasks, db, force_update, start_date
    )


@router.post("/bulk-incremental-update")
@api_safe_execute(message="一括差分データ更新エラー")
async def update_bulk_incremental_data(
    symbol: str = DEFAULT_MARKET_SYMBOL,
    db: Session = Depends(get_db),
    orchestration_service: DataCollectionOrchestrationService = Depends(
        get_data_collection_orchestration_service
    ),
) -> Dict:
    """
    一括差分データを更新（OHLCV、FR、OI）
    """
    return await orchestration_service.execute_bulk_incremental_update(symbol, db)


@router.post("/bulk-historical")
@api_safe_execute(message="一括履歴データ収集エラー")
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
    """
    return await orchestration_service.start_bulk_historical_data_collection(
        background_tasks, db, force_update, start_date
    )


@router.get("/status/{symbol:path}/{timeframe}")
@api_safe_execute(message="データ収集状況確認エラー")
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
    """
    return await orchestration_service.get_collection_status(
        symbol=symbol,
        timeframe=timeframe,
        background_tasks=background_tasks,
        auto_fetch=auto_fetch,
        db=db,
    )


@router.post("/all/bulk-collect")
@api_safe_execute(message="全データ一括収集エラー")
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
    """
    # 全データ一括収集サービスにも上書きオプションを追加する必要があるため、
    # 一旦はbulk-historicalを使用する
    return await orchestration_service.start_bulk_historical_data_collection(
        background_tasks, db, force_update, start_date
    )


@router.post("/historical-oi")
@api_safe_execute(message="OIデータ履歴収集エラー")
async def collect_historical_oi_data(
    background_tasks: BackgroundTasks,
    symbol: str = DEFAULT_MARKET_SYMBOL,
    interval: str = "1h",
    db: Session = Depends(get_db),
    orchestration_service: DataCollectionOrchestrationService = Depends(
        get_data_collection_orchestration_service
    ),
) -> Dict:
    """
    OI（Open Interest）履歴データを収集
    """
    return await orchestration_service.start_historical_oi_collection(
        symbol, interval, background_tasks, db
    )
