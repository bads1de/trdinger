"""
Fear & Greed Index API エンドポイント

Alternative.me Fear & Greed Index データの取得、収集、管理機能を提供します。
責務の分離により、ビジネスロジックはサービス層に委譲されています。
"""

import logging
from typing import Dict, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.services.data_collection.orchestration.fear_greed_orchestration_service import (
    FearGreedOrchestrationService,
)
from app.utils.unified_error_handler import UnifiedErrorHandler
from database.connection import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/fear-greed", tags=["fear-greed"])


@router.get("/data")
async def get_fear_greed_data(
    start_date: Optional[str] = Query(None, description="開始日時 (ISO format)"),
    end_date: Optional[str] = Query(None, description="終了日時 (ISO format)"),
    limit: Optional[int] = Query(30, description="取得件数制限"),
    db: Session = Depends(get_db),
    service: FearGreedOrchestrationService = Depends(FearGreedOrchestrationService),
) -> Dict:
    """
    Fear & Greed Index データを取得

    Args:
        start_date: 開始日時（ISO形式）
        end_date: 終了日時（ISO形式）
        limit: 取得件数制限
        db: データベースセッション
        service: Fear & Greedオーケストレーションサービス（依存性注入）

    Returns:
        Fear & Greed Index データ
    """

    async def _get_fear_greed_data():
        return await service.get_fear_greed_data(
            db=db, start_date=start_date, end_date=end_date, limit=limit
        )

    return await UnifiedErrorHandler.safe_execute_async(_get_fear_greed_data)


@router.get("/latest")
async def get_latest_fear_greed_data(
    limit: int = Query(30, description="取得件数制限"),
    db: Session = Depends(get_db),
    service: FearGreedOrchestrationService = Depends(FearGreedOrchestrationService),
) -> Dict:
    """
    最新のFear & Greed Index データを取得

    Args:
        limit: 取得件数制限
        db: データベースセッション
        service: Fear & Greedオーケストレーションサービス（依存性注入）

    Returns:
        最新のFear & Greed Index データ
    """

    async def _get_latest_data():
        return await service.get_latest_fear_greed_data(db=db, limit=limit)

    return await UnifiedErrorHandler.safe_execute_async(_get_latest_data)


@router.get("/status")
async def get_fear_greed_data_status(
    db: Session = Depends(get_db),
) -> Dict:
    """
    Fear & Greed Index データの状態を取得

    Args:
        db: データベースセッション

    Returns:
        データ状態情報
    """

    async def _execute():
        orchestration_service = FearGreedOrchestrationService()
        return await orchestration_service.get_fear_greed_data_status(db)

    return await UnifiedErrorHandler.safe_execute_async(_execute)


@router.post("/collect")
async def collect_fear_greed_data(
    limit: int = Query(30, description="取得するデータ数"),
    db: Session = Depends(get_db),
    orchestration_service: FearGreedOrchestrationService = Depends(
        FearGreedOrchestrationService
    ),
) -> Dict:
    """
    Fear & Greed Index データを収集

    Args:
        limit: 取得するデータ数
        db: データベースセッション
        orchestration_service: Fear & Greedオーケストレーションサービス（依存性注入）

    Returns:
        収集結果
    """

    async def _execute():
        return await orchestration_service.collect_fear_greed_data(limit, db)

    return await UnifiedErrorHandler.safe_execute_async(_execute)


@router.post("/collect-incremental")
async def collect_incremental_fear_greed_data(
    db: Session = Depends(get_db),
) -> Dict:
    """
    Fear & Greed Index の差分データを収集

    Args:
        db: データベースセッション

    Returns:
        差分収集結果
    """

    async def _execute():

        orchestration_service = FearGreedOrchestrationService()
        return await orchestration_service.collect_incremental_fear_greed_data(db)

    return await UnifiedErrorHandler.safe_execute_async(_execute)


@router.post("/collect-historical")
async def collect_historical_fear_greed_data(
    limit: int = Query(1000, description="取得するデータ数の上限"),
    db: Session = Depends(get_db),
) -> Dict:
    """
    Fear & Greed Index の履歴データを収集

    Args:
        limit: 取得するデータ数の上限
        db: データベースセッション

    Returns:
        履歴収集結果
    """

    async def _execute():
        orchestration_service = FearGreedOrchestrationService()
        return await orchestration_service.collect_historical_fear_greed_data(limit, db)

    return await UnifiedErrorHandler.safe_execute_async(_execute)


@router.delete("/cleanup")
async def cleanup_old_fear_greed_data(
    days_to_keep: int = Query(365, description="保持する日数"),
    db: Session = Depends(get_db),
) -> Dict:
    """
    古いFear & Greed Index データをクリーンアップ

    Args:
        days_to_keep: 保持する日数
        db: データベースセッション

    Returns:
        クリーンアップ結果
    """

    async def _execute():
        orchestration_service = FearGreedOrchestrationService()
        return await orchestration_service.cleanup_old_fear_greed_data(days_to_keep, db)

    return await UnifiedErrorHandler.safe_execute_async(_execute)
