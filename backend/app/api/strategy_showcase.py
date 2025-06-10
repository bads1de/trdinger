"""
戦略ショーケースAPI

自動生成された投資戦略のショーケース機能を提供するAPIエンドポイント
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import logging

from app.core.services.strategy_showcase_service import StrategyShowcaseService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/strategies", tags=["strategy-showcase"])

# グローバルサービスインスタンス
showcase_service = StrategyShowcaseService()


# リクエスト・レスポンスモデル


class GenerateShowcaseRequest(BaseModel):
    """ショーケース戦略生成リクエスト"""

    count: int = Field(default=30, ge=1, le=50, description="生成する戦略数")
    base_config: Optional[Dict[str, Any]] = Field(
        None, description="基本バックテスト設定"
    )

    class Config:
        schema_extra = {
            "example": {
                "count": 30,
                "base_config": {
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-19",
                    "initial_capital": 100000,
                    "commission_rate": 0.00055,
                },
            }
        }


class GenerateShowcaseResponse(BaseModel):
    """ショーケース戦略生成レスポンス"""

    success: bool
    message: str
    generated_count: int
    saved_count: int


class StrategyListResponse(BaseModel):
    """戦略一覧レスポンス"""

    success: bool
    strategies: List[Dict[str, Any]]
    total_count: int
    message: str


class StrategyDetailResponse(BaseModel):
    """戦略詳細レスポンス"""

    success: bool
    strategy: Optional[Dict[str, Any]] = None
    message: str


class ShowcaseStatsResponse(BaseModel):
    """ショーケース統計レスポンス"""

    success: bool
    statistics: Dict[str, Any]
    message: str


# APIエンドポイント


@router.post("/showcase/generate", response_model=GenerateShowcaseResponse)
async def generate_showcase_strategies(
    request: GenerateShowcaseRequest, background_tasks: BackgroundTasks
):
    """
    ショーケース用戦略を生成

    30個の多様な投資戦略を自動生成し、データベースに保存します。
    バックグラウンドで実行されるため、即座にレスポンスを返します。
    """
    try:
        logger.info(f"ショーケース戦略生成開始: {request.count}個")

        # バックグラウンドタスクとして実行
        background_tasks.add_task(
            _generate_and_save_strategies, request.count, request.base_config
        )

        return GenerateShowcaseResponse(
            success=True,
            message=f"ショーケース戦略の生成を開始しました。{request.count}個の戦略を生成中です。",
            generated_count=0,  # バックグラウンド実行のため初期値
            saved_count=0,
        )

    except Exception as e:
        logger.error(f"ショーケース戦略生成エラー: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/showcase", response_model=StrategyListResponse)
async def get_showcase_strategies(
    category: Optional[str] = Query(None, description="戦略カテゴリでフィルタ"),
    risk_level: Optional[str] = Query(None, description="リスクレベルでフィルタ"),
    limit: Optional[int] = Query(None, ge=1, le=100, description="取得件数制限"),
    offset: int = Query(0, ge=0, description="オフセット"),
    sort_by: str = Query("expected_return", description="ソート項目"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="ソート順序"),
):
    """
    ショーケース戦略一覧を取得

    フィルタリング、ソート、ページネーション機能付きで戦略一覧を取得します。
    """
    try:
        strategies = showcase_service.get_showcase_strategies(
            category=category,
            risk_level=risk_level,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        return StrategyListResponse(
            success=True,
            strategies=strategies,
            total_count=len(strategies),
            message="戦略一覧を取得しました",
        )

    except Exception as e:
        logger.error(f"戦略一覧取得エラー: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/showcase/stats", response_model=ShowcaseStatsResponse)
async def get_showcase_statistics():
    """
    ショーケース統計情報を取得

    全戦略の統計情報（平均リターン、カテゴリ分布等）を取得します。
    """
    try:
        statistics = showcase_service.get_showcase_statistics()

        return ShowcaseStatsResponse(
            success=True, statistics=statistics, message="統計情報を取得しました"
        )

    except Exception as e:
        logger.error(f"統計情報取得エラー: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/showcase/{strategy_id}", response_model=StrategyDetailResponse)
async def get_strategy_detail(strategy_id: int):
    """
    戦略詳細を取得

    指定されたIDの戦略の詳細情報を取得します。
    """
    try:
        strategy = showcase_service.get_strategy_by_id(strategy_id)

        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")

        return StrategyDetailResponse(
            success=True, strategy=strategy, message="戦略詳細を取得しました"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"戦略詳細取得エラー: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/categories")
async def get_strategy_categories():
    """
    利用可能な戦略カテゴリ一覧を取得
    """
    try:
        categories = {
            "trend_following": "トレンドフォロー",
            "mean_reversion": "逆張り",
            "breakout": "ブレイクアウト",
            "range_trading": "レンジ取引",
            "momentum": "モメンタム",
        }

        return {
            "success": True,
            "categories": categories,
            "message": "戦略カテゴリ一覧を取得しました",
        }

    except Exception as e:
        logger.error(f"カテゴリ一覧取得エラー: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/risk-levels")
async def get_risk_levels():
    """
    利用可能なリスクレベル一覧を取得
    """
    try:
        risk_levels = {"low": "低リスク", "medium": "中リスク", "high": "高リスク"}

        return {
            "success": True,
            "risk_levels": risk_levels,
            "message": "リスクレベル一覧を取得しました",
        }

    except Exception as e:
        logger.error(f"リスクレベル一覧取得エラー: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ヘルパー関数


async def _generate_and_save_strategies(
    count: int, base_config: Optional[Dict[str, Any]]
):
    """
    戦略生成と保存をバックグラウンドで実行

    Args:
        count: 生成する戦略数
        base_config: 基本バックテスト設定
    """
    try:
        logger.info(f"バックグラウンド戦略生成開始: {count}個")

        # 戦略を生成
        strategies = showcase_service.generate_showcase_strategies(count, base_config)

        # データベースに保存
        saved_ids = showcase_service.save_strategies_to_database(strategies)

        logger.info(
            f"バックグラウンド戦略生成完了: {len(strategies)}個生成, {len(saved_ids)}個保存"
        )

    except Exception as e:
        logger.error(f"バックグラウンド戦略生成エラー: {e}")
