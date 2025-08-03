"""
バックテストAPIエンドポイント

backtesting.pyライブラリを使用したバックテスト機能のAPIを提供します。
責務の分離により、ビジネスロジックはOrchestrationServiceに委譲されています。
"""

from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query

from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from database.connection import get_db
from app.services.backtest.orchestration.backtest_orchestration_service import (
    BacktestOrchestrationService,
)
from app.utils.unified_error_handler import UnifiedErrorHandler

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


# Pydanticモデル定義
class StrategyConfig(BaseModel):
    """戦略設定"""

    strategy_type: str = Field(..., description="戦略タイプ")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="戦略パラメータ"
    )


class BacktestRequest(BaseModel):
    """バックテストリクエスト"""

    strategy_name: str = Field(..., description="戦略名")
    symbol: str = Field(..., description="取引ペア")
    timeframe: str = Field(..., description="時間軸")
    start_date: datetime = Field(..., description="開始日時")
    end_date: datetime = Field(..., description="終了日時")
    initial_capital: float = Field(..., gt=0, description="初期資金")
    commission_rate: float = Field(default=0.00055, ge=0, le=1, description="手数料率")
    strategy_config: StrategyConfig = Field(..., description="戦略設定")


class BacktestResponse(BaseModel):
    """バックテストレスポンス"""

    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BacktestResultsResponse(BaseModel):
    """バックテスト結果一覧レスポンス"""

    success: bool
    results: Optional[list] = None
    total: Optional[int] = None
    error: Optional[str] = None


@router.get("/results", response_model=BacktestResultsResponse)
async def get_backtest_results(
    limit: int = Query(50, ge=1, le=100, description="取得件数"),
    offset: int = Query(0, ge=0, description="オフセット"),
    symbol: Optional[str] = Query(None, description="取引ペアフィルター"),
    strategy_name: Optional[str] = Query(None, description="戦略名フィルター"),
    db: Session = Depends(get_db),
    orchestration_service: BacktestOrchestrationService = Depends(
        BacktestOrchestrationService
    ),
):
    """
    バックテスト結果一覧を取得

    Args:
        limit: 取得件数
        offset: オフセット
        symbol: 取引ペアフィルター
        strategy_name: 戦略名フィルター
        db: データベースセッション
        orchestration_service: バックテストオーケストレーションサービス（依存性注入）

    Returns:
        バックテスト結果一覧
    """

    async def _get_results():
        return await orchestration_service.get_backtest_results(
            db=db,
            limit=limit,
            offset=offset,
            symbol=symbol,
            strategy_name=strategy_name,
        )

    return await UnifiedErrorHandler.safe_execute_async(_get_results)


@router.delete("/results-all")
async def delete_all_backtest_results(
    db: Session = Depends(get_db),
    orchestration_service: BacktestOrchestrationService = Depends(
        BacktestOrchestrationService
    ),
):
    """
    すべてのバックテスト結果を削除

    Args:
        db: データベースセッション
        orchestration_service: バックテストオーケストレーションサービス（依存性注入）

    Returns:
        削除結果
    """

    async def _delete_all_results():
        return await orchestration_service.delete_all_backtest_results(db=db)

    return await UnifiedErrorHandler.safe_execute_async(_delete_all_results)


@router.get("/results/{result_id}", response_model=BacktestResponse)
async def get_backtest_result_by_id(
    result_id: int,
    db: Session = Depends(get_db),
    orchestration_service: BacktestOrchestrationService = Depends(
        BacktestOrchestrationService
    ),
):
    """
    ID指定でバックテスト結果を取得

    Args:
        result_id: バックテスト結果ID
        db: データベースセッション
        orchestration_service: バックテストオーケストレーションサービス（依存性注入）

    Returns:
        バックテスト結果
    """

    async def _get_by_id():
        result = await orchestration_service.get_backtest_result_by_id(
            db=db, result_id=result_id
        )

        # エラーハンドリング
        if not result["success"]:
            raise HTTPException(
                status_code=result.get("status_code", 500),
                detail=result.get("error", "Unknown error"),
            )

        return result

    return await UnifiedErrorHandler.safe_execute_async(_get_by_id)


@router.delete("/results/{result_id}")
async def delete_backtest_result(
    result_id: int,
    db: Session = Depends(get_db),
    orchestration_service: BacktestOrchestrationService = Depends(
        BacktestOrchestrationService
    ),
):
    """
    バックテスト結果を削除

    Args:
        result_id: バックテスト結果ID
        db: データベースセッション
        orchestration_service: バックテストオーケストレーションサービス（依存性注入）

    Returns:
        削除結果
    """

    async def _delete_result():
        result = await orchestration_service.delete_backtest_result(
            db=db, result_id=result_id
        )

        # エラーハンドリング
        if not result["success"]:
            raise HTTPException(
                status_code=result.get("status_code", 500),
                detail=result.get("error", "Unknown error"),
            )

        return result

    return await UnifiedErrorHandler.safe_execute_async(_delete_result)


@router.get("/strategies")
async def get_supported_strategies(
    orchestration_service: BacktestOrchestrationService = Depends(
        BacktestOrchestrationService
    ),
):
    """
    サポートされている戦略一覧を取得

    Args:
        orchestration_service: バックテストオーケストレーションサービス（依存性注入）

    Returns:
        戦略一覧
    """

    async def _get_strategies():
        return await orchestration_service.get_supported_strategies()

    return await UnifiedErrorHandler.safe_execute_async(_get_strategies)
