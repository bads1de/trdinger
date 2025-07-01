"""
統合戦略API

ショーケース戦略とオートストラテジー由来の戦略を統合して提供するAPIエンドポイントです。
"""

import logging

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from app.core.services.strategy_integration_service import StrategyIntegrationService
from database.connection import get_db
from app.core.utils.api_utils import APIResponseHelper, APIErrorHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/strategies", tags=["strategies"])


# レスポンスモデル
class UnifiedStrategiesResponse(BaseModel):
    """統合戦略レスポンス"""

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


@router.get("/unified", response_model=UnifiedStrategiesResponse)
async def get_unified_strategies(
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
    統合された戦略一覧を取得

    ショーケース戦略とオートストラテジー由来の戦略を統合して返します。

    Args:
        limit: 取得件数制限 (1-100)
        offset: オフセット
        category: カテゴリフィルター
        risk_level: リスクレベルフィルター (low, medium, high)
        sort_by: ソート項目 (created_at, expected_return, sharpe_ratio, max_drawdown, win_rate)
        sort_order: ソート順序 (asc, desc)
        db: データベースセッション

    Returns:
        統合された戦略データ
    """

    async def _get_unified():
        logger.info(
            f"統合戦略取得開始: limit={limit}, offset={offset}, category={category}"
        )

        integration_service = StrategyIntegrationService(db)

        result = integration_service.get_unified_strategies(
            limit=limit,
            offset=offset,
            category=category,
            risk_level=risk_level,
            experiment_id=experiment_id,
            min_fitness=min_fitness,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        logger.info(f"統合戦略取得完了: {len(result['strategies'])} 件")

        return APIResponseHelper.api_response(
            success=True,
            data={
                "strategies": result["strategies"],
                "total_count": result["total_count"],
                "has_more": result["has_more"],
            },
            message="統合戦略を正常に取得しました",
        )

    return await APIErrorHandler.handle_api_exception(_get_unified)


@router.get("/auto-generated", response_model=UnifiedStrategiesResponse)
async def get_auto_generated_strategies(
    limit: int = Query(20, ge=1, le=100, description="取得件数制限"),
    offset: int = Query(0, ge=0, description="オフセット"),
    experiment_id: Optional[int] = Query(None, description="実験IDフィルター"),
    min_fitness: Optional[float] = Query(None, description="最小フィットネススコア"),
    sort_by: str = Query("fitness_score", description="ソート項目"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="ソート順序"),
    db: Session = Depends(get_db),
):
    """
    オートストラテジー由来の戦略のみを取得

    Args:
        limit: 取得件数制限
        offset: オフセット
        experiment_id: 実験IDフィルター
        min_fitness: 最小フィットネススコア
        sort_by: ソート項目
        sort_order: ソート順序
        db: データベースセッション

    Returns:
        オートストラテジー戦略データ
    """

    async def _get_auto_strategies():
        logger.info(f"オートストラテジー取得開始: experiment_id={experiment_id}")

        integration_service = StrategyIntegrationService(db)

        auto_strategies = integration_service._get_auto_generated_strategies(
            limit=limit, offset=offset, sort_by=sort_by, sort_order=sort_order
        )

        if experiment_id is not None:
            auto_strategies = [
                s for s in auto_strategies if s.get("experiment_id") == experiment_id
            ]

        if min_fitness is not None:
            auto_strategies = [
                s for s in auto_strategies if s.get("fitness_score", 0) >= min_fitness
            ]

        total_count = len(auto_strategies)
        paginated_strategies = auto_strategies[offset : offset + limit]

        logger.info(f"オートストラテジー取得完了: {len(paginated_strategies)} 件")

        return APIResponseHelper.api_response(
            success=True,
            data={
                "strategies": paginated_strategies,
                "total_count": total_count,
                "has_more": offset + limit < total_count,
            },
            message="自動生成戦略を正常に取得しました",
        )

    return await APIErrorHandler.handle_api_exception(_get_auto_strategies)


@router.get("/stats", response_model=StrategyStatsResponse)
async def get_strategy_statistics(db: Session = Depends(get_db)):
    """
    戦略統計情報を取得

    Args:
        db: データベースセッション

    Returns:
        戦略統計データ
    """

    async def _get_stats():
        logger.info("戦略統計取得開始")

        integration_service = StrategyIntegrationService(db)

        all_strategies_result = integration_service.get_unified_strategies(
            limit=1000, offset=0
        )

        strategies = all_strategies_result["strategies"]

        stats = {
            "total_strategies": len(strategies),
            "auto_generated_strategies": len(
                [s for s in strategies if s["source"] == "auto_strategy"]
            ),
            "categories": {},
            "risk_levels": {},
            "performance_summary": {
                "avg_return": 0.0,
                "avg_sharpe_ratio": 0.0,
                "avg_max_drawdown": 0.0,
                "avg_win_rate": 0.0,
            },
        }

        for strategy in strategies:
            category = strategy.get("category", "unknown")
            stats["categories"][category] = stats["categories"].get(category, 0) + 1

            risk_level = strategy.get("risk_level", "unknown")
            stats["risk_levels"][risk_level] = (
                stats["risk_levels"].get(risk_level, 0) + 1
            )

        if strategies:
            stats["performance_summary"]["avg_return"] = sum(
                s.get("expected_return", 0) for s in strategies
            ) / len(strategies)

            stats["performance_summary"]["avg_sharpe_ratio"] = sum(
                s.get("sharpe_ratio", 0) for s in strategies
            ) / len(strategies)

            stats["performance_summary"]["avg_max_drawdown"] = sum(
                s.get("max_drawdown", 0) for s in strategies
            ) / len(strategies)

            stats["performance_summary"]["avg_win_rate"] = sum(
                s.get("win_rate", 0) for s in strategies
            ) / len(strategies)

        logger.info("戦略統計取得完了")

        return APIResponseHelper.api_response(
            data=stats, message="戦略統計情報を正常に取得しました", success=True
        )

    return await APIErrorHandler.handle_api_exception(_get_stats)


@router.get("/categories")
async def get_strategy_categories():
    """
    利用可能な戦略カテゴリ一覧を取得

    Returns:
        カテゴリ一覧
    """
    try:
        categories = [
            {"value": "trend_following", "label": "トレンドフォロー"},
            {"value": "mean_reversion", "label": "平均回帰"},
            {"value": "momentum", "label": "モメンタム"},
            {"value": "breakout", "label": "ブレイクアウト"},
            {"value": "scalping", "label": "スキャルピング"},
            {"value": "swing", "label": "スイング"},
            {"value": "auto_generated", "label": "自動生成"},
            {"value": "hybrid", "label": "ハイブリッド"},
        ]

        return {
            "success": True,
            "categories": categories,
            "message": "カテゴリが正常に取得されました",
        }

    except Exception as e:
        logger.error(f"カテゴリ取得中にエラーが発生しました: {e}")
        raise HTTPException(
            status_code=500, detail=f"カテゴリの取得に失敗しました: {str(e)}"
        )


@router.get("/risk-levels")
async def get_risk_levels():
    """
    利用可能なリスクレベル一覧を取得

    Returns:
        リスクレベル一覧
    """
    try:
        risk_levels = [
            {
                "value": "low",
                "label": "低リスク",
                "description": "最大ドローダウン 5%以下",
            },
            {
                "value": "medium",
                "label": "中リスク",
                "description": "最大ドローダウン 5-15%",
            },
            {
                "value": "high",
                "label": "高リスク",
                "description": "最大ドローダウン 15%以上",
            },
        ]

        return {
            "success": True,
            "risk_levels": risk_levels,
            "message": "リスクレベルが正常に取得されました",
        }

    except Exception as e:
        logger.error(f"リスクレベル取得中にエラーが発生しました: {e}")
        raise HTTPException(
            status_code=500, detail=f"リスクレベルの取得に失敗しました: {str(e)}"
        )
