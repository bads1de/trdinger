"""
バックテストAPIエンドポイント

backtesting.pyライブラリを使用したバックテスト機能のAPIを提供します。
"""

from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from database.connection import get_db
from database.repositories.backtest_result_repository import BacktestResultRepository
from app.core.services.backtest_service import BacktestService
from app.core.utils.api_utils import APIErrorHandler
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


class EnhancedOptimizationRequest(BaseModel):
    """拡張最適化リクエスト"""

    base_config: BacktestRequest = Field(..., description="基本設定")
    optimization_params: Dict[str, Any] = Field(..., description="最適化パラメータ")


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
def _create_base_config(request: BacktestRequest) -> Dict[str, Any]:
    """バックテストリクエストから基本設定辞書を作成"""
    return {
        "strategy_name": request.strategy_name,
        "symbol": request.symbol,
        "timeframe": request.timeframe,
        "start_date": request.start_date,
        "end_date": request.end_date,
        "initial_capital": request.initial_capital,
        "commission_rate": request.commission_rate,
        "strategy_config": {
            "strategy_type": request.strategy_config.strategy_type,
            "parameters": request.strategy_config.parameters,
        },
    }


def _save_backtest_result(result: Dict[str, Any], db: Session) -> Dict[str, Any]:
    """バックテスト結果をデータベースに保存"""
    backtest_repo = BacktestResultRepository(db)
    return backtest_repo.save_backtest_result(result)


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

    async def _run():
        backtest_service = BacktestService()
        config = _create_base_config(request)
        result = backtest_service.run_backtest(config)
        saved_result = _save_backtest_result(result, db)
        return {"success": True, "result": saved_result}

    return await APIErrorHandler.handle_api_exception(_run)


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

    return await APIErrorHandler.handle_api_exception(_get_results)


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

    return await APIErrorHandler.handle_api_exception(_delete_all_results)


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

    return await APIErrorHandler.handle_api_exception(_get_by_id)


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

    return await APIErrorHandler.handle_api_exception(_delete_result)


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

    return await APIErrorHandler.handle_api_exception(_get_strategies)


@router.post("/optimize-enhanced", response_model=BacktestResponse)
async def optimize_strategy_enhanced(
    request: EnhancedOptimizationRequest, db: Session = Depends(get_db)
):
    """
    戦略を拡張最適化

    Args:
        request: 拡張最適化リクエスト
        db: データベースセッション

    Returns:
        最適化結果
    """

    async def _optimize_enhanced():
        backtest_service = BacktestService()
        base_config = _create_base_config(request.base_config)
        result = backtest_service.optimize_strategy_enhanced(
            base_config, request.optimization_params
        )
        saved_result = _save_backtest_result(result, db)
        return {"success": True, "result": saved_result}

    return await APIErrorHandler.handle_api_exception(_optimize_enhanced)
