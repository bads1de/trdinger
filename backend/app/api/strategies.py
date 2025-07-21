"""
オートストラテジー戦略API

オートストラテジー由来の戦略を提供するAPIエンドポイントです。
"""

import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from app.core.dependencies import get_strategy_integration_service
from database.connection import get_db
from app.core.utils.unified_error_handler import UnifiedErrorHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/strategies", tags=["strategies"])


# レスポンスモデル
class StrategiesResponse(BaseModel):
    """戦略レスポンス"""

    success: bool = True
    strategies: list = Field(default_factory=list)
    total_count: int = 0
    has_more: bool = False
    message: str = "戦略が正常に取得されました"
    timestamp: str = Field(
        default_factory=lambda: __import__("datetime").datetime.now().isoformat()
    )


class StrategyStatsResponse(BaseModel):
    """戦略統計レスポンス"""

    success: bool = True
    stats: Dict[str, Any] = Field(default_factory=dict)
    message: str = "戦略統計が正常に取得されました"
    timestamp: str = Field(
        default_factory=lambda: __import__("datetime").datetime.now().isoformat()
    )


@router.get("/", response_model=StrategiesResponse)
async def get_strategies(
    limit: int = Query(20, ge=1, le=100, description="取得件数制限"),
    offset: int = Query(0, ge=0, description="オフセット"),
    category: Optional[str] = Query(None, description="カテゴリフィルター"),
    risk_level: Optional[str] = Query(None, description="リスクレベルフィルター"),
    experiment_id: Optional[int] = Query(None, description="実験IDフィルター"),
    min_fitness: Optional[float] = Query(None, description="最小フィットネススコア"),
    sort_by: str = Query("fitness_score", description="ソート項目"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="ソート順序"),
    db: Session = Depends(get_db),
):
    """
    生成された戦略の一覧を取得

    オートストラテジーによって生成された戦略をフィルタリング、ソート、
    ページネーションして返します。

    Args:
        limit: 取得件数制限 (1-100)
        offset: オフセット
        category: カテゴリフィルター
        risk_level: リスクレベルフィルター (low, medium, high)
        experiment_id: 実験IDフィルター
        min_fitness: 最小フィットネススコア
        sort_by: ソート項目 (fitness_score, created_at, expected_return, sharpe_ratio, max_drawdown, win_rate)
        sort_order: ソート順序 (asc, desc)
        db: データベースセッション

    Returns:
        生成された戦略データ
    """

    # ビジネスロジックをサービス層に委譲
    service = get_strategy_integration_service(db)

    async def _get_strategies():
        return service.get_strategies_with_response(
            limit=limit,
            offset=offset,
            category=category,
            risk_level=risk_level,
            experiment_id=experiment_id,
            min_fitness=min_fitness,
            sort_by=sort_by,
            sort_order=sort_order,
        )

    return await UnifiedErrorHandler.safe_execute_async(_get_strategies)
