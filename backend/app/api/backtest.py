"""
バックテストAPIエンドポイント

backtesting.pyライブラリを使用したバックテスト機能のAPIを提供します。
"""

from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from database.connection import get_db
from database.repositories.backtest_result_repository import BacktestResultRepository
from app.core.services.backtest_service import BacktestService
from app.core.utils.unified_error_handler import UnifiedErrorHandler
from app.core.dependencies import get_backtest_service_with_db
from database.repositories.ga_experiment_repository import GAExperimentRepository
from database.repositories.generated_strategy_repository import (
    GeneratedStrategyRepository,
)

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


# ヘルパー関数
# 以下のヘルパー関数は廃止予定（サービス層に移行済み）
# def _create_base_config(request: BacktestRequest) -> Dict[str, Any]:
# def _save_backtest_result(result: Dict[str, Any], db: Session) -> Dict[str, Any]:


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest, db: Session = Depends(get_db)):
    """
    バックテストを実行

    Args:
        request: バックテストリクエスト
        db: データベースセッション

    Returns:
        バックテスト結果
    """

    async def _run_backtest():
        # ビジネスロジックをサービス層に委譲
        backtest_service = get_backtest_service_with_db(db)

        # 同期関数をスレッドプールで実行
        result = await run_in_threadpool(
            backtest_service.execute_and_save_backtest, request, db
        )
        return result

    return await UnifiedErrorHandler.safe_execute_async(
        _run_backtest, message="バックテストの実行エラー"
    )


@router.get("/results", response_model=BacktestResultsResponse)
async def get_backtest_results(
    limit: int = Query(50, ge=1, le=100, description="取得件数"),
    offset: int = Query(0, ge=0, description="オフセット"),
    symbol: Optional[str] = Query(None, description="取引ペアフィルター"),
    strategy_name: Optional[str] = Query(None, description="戦略名フィルター"),
    db: Session = Depends(get_db),
):
    """
    バックテスト結果一覧を取得

    Args:
        limit: 取得件数
        offset: オフセット
        symbol: 取引ペアフィルター
        strategy_name: 戦略名フィルター
        db: データベースセッション

    Returns:
        バックテスト結果一覧
    """

    async def _get_results():
        backtest_repo = BacktestResultRepository(db)

        results = backtest_repo.get_backtest_results(
            limit=limit, offset=offset, symbol=symbol, strategy_name=strategy_name
        )

        total = backtest_repo.count_backtest_results(
            symbol=symbol, strategy_name=strategy_name
        )

        return {"success": True, "results": results, "total": total}

    return await UnifiedErrorHandler.safe_execute_async(_get_results)


@router.delete("/results-all")
async def delete_all_backtest_results(db: Session = Depends(get_db)):
    """
    すべてのバックテスト結果を削除

    Args:
        db: データベースセッション

    Returns:
        削除結果
    """

    async def _delete_all_results():
        backtest_repo = BacktestResultRepository(db)
        ga_experiment_repo = GAExperimentRepository(db)
        generated_strategy_repo = GeneratedStrategyRepository(db)

        # 関連テーブルのデータをすべて削除
        deleted_strategies_count = generated_strategy_repo.delete_all_strategies()
        deleted_experiments_count = ga_experiment_repo.delete_all_experiments()
        deleted_backtests_count = backtest_repo.delete_all_backtest_results()

        total_deleted = (
            deleted_strategies_count
            + deleted_experiments_count
            + deleted_backtests_count
        )

        return {
            "success": True,
            "message": f"All related data deleted successfully ({total_deleted} records)",
            "deleted_counts": {
                "backtest_results": deleted_backtests_count,
                "ga_experiments": deleted_experiments_count,
                "generated_strategies": deleted_strategies_count,
            },
        }

    return await UnifiedErrorHandler.safe_execute_async(_delete_all_results)


@router.get("/results/{result_id}", response_model=BacktestResponse)
async def get_backtest_result_by_id(result_id: int, db: Session = Depends(get_db)):
    """
    ID指定でバックテスト結果を取得

    Args:
        result_id: バックテスト結果ID
        db: データベースセッション

    Returns:
        バックテスト結果
    """

    async def _get_by_id():
        backtest_repo = BacktestResultRepository(db)
        result = backtest_repo.get_backtest_result_by_id(result_id)

        if result is None:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        return {"success": True, "result": result}

    return await UnifiedErrorHandler.safe_execute_async(_get_by_id)


@router.delete("/results/{result_id}")
async def delete_backtest_result(result_id: int, db: Session = Depends(get_db)):
    """
    バックテスト結果を削除

    Args:
        result_id: バックテスト結果ID
        db: データベースセッション

    Returns:
        削除結果
    """

    async def _delete_result():
        backtest_repo = BacktestResultRepository(db)
        success = backtest_repo.delete_backtest_result(result_id)

        if not success:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        return {"success": True, "message": "Backtest result deleted successfully"}

    return await UnifiedErrorHandler.safe_execute_async(_delete_result)


@router.get("/strategies")
async def get_supported_strategies():
    """
    サポートされている戦略一覧を取得

    Returns:
        戦略一覧
    """

    async def _get_strategies():
        backtest_service = BacktestService()
        strategies = backtest_service.get_supported_strategies()
        return {"success": True, "strategies": strategies}

    return await UnifiedErrorHandler.safe_execute_async(_get_strategies)
