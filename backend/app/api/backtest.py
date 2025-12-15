"""
バックテストAPIエンドポイント

backtesting.pyライブラリを使用したバックテスト機能のAPIを提供します。
責務の分離により、ビジネスロジックはOrchestrationServiceに委譲されています。
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.dependencies import get_backtest_orchestration_service
from app.config.unified_config import unified_config
from app.services.backtest.orchestration.backtest_orchestration_service import (
    BacktestOrchestrationService,
)
from app.utils.error_handler import ErrorHandler
from database.connection import get_db

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


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
    limit: int = Query(
        unified_config.backtest.default_results_limit,
        ge=1,
        le=unified_config.backtest.max_results_limit,
        description="取得件数",
    ),
    offset: int = Query(0, ge=0, description="オフセット"),
    symbol: Optional[str] = Query(None, description="取引ペアフィルター"),
    strategy_name: Optional[str] = Query(None, description="戦略名フィルター"),
    db: Session = Depends(get_db),
    orchestration_service: BacktestOrchestrationService = Depends(
        get_backtest_orchestration_service
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
        # orchestration_service returns api_response with data field: {"results": [...], "total": N}
        resp = await orchestration_service.get_backtest_results(
            db=db,
            limit=limit,
            offset=offset,
            symbol=symbol,
            strategy_name=strategy_name,
        )

        # orchestration_service now returns normalized response with top-level `results` and `total`
        return resp

    return await ErrorHandler.safe_execute_async(_get_results)


@router.delete("/results-all")
async def delete_all_backtest_results(
    db: Session = Depends(get_db),
    orchestration_service: BacktestOrchestrationService = Depends(
        get_backtest_orchestration_service
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

    return await ErrorHandler.safe_execute_async(_delete_all_results)


@router.get("/results/{result_id}/", response_model=BacktestResponse)
async def get_backtest_result_by_id(
    result_id: int,
    db: Session = Depends(get_db),
    orchestration_service: BacktestOrchestrationService = Depends(
        get_backtest_orchestration_service
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

    return await ErrorHandler.safe_execute_async(_get_by_id)


@router.delete("/results/{result_id}/")
async def delete_backtest_result(
    result_id: int,
    db: Session = Depends(get_db),
    orchestration_service: BacktestOrchestrationService = Depends(
        get_backtest_orchestration_service
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

    return await ErrorHandler.safe_execute_async(_delete_result)


@router.get("/strategies")
async def get_supported_strategies(
    orchestration_service: BacktestOrchestrationService = Depends(
        get_backtest_orchestration_service
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

    return await ErrorHandler.safe_execute_async(_get_strategies)


