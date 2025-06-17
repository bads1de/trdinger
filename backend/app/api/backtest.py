"""
バックテストAPIエンドポイント

backtesting.pyライブラリを使用したバックテスト機能のAPIを提供します。
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from database.connection import get_db
from database.repositories.backtest_result_repository import BacktestResultRepository
from app.core.services.backtest_service import BacktestService
from app.core.services.enhanced_backtest_service import EnhancedBacktestService
from app.utils.api_response_utils import handle_api_exception

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


class OptimizationRequest(BaseModel):
    """最適化リクエスト"""

    base_config: BacktestRequest = Field(..., description="基本設定")
    optimization_params: Dict[str, Any] = Field(..., description="最適化パラメータ")


class EnhancedOptimizationRequest(BaseModel):
    """拡張最適化リクエスト"""

    base_config: BacktestRequest = Field(..., description="基本設定")
    optimization_params: Dict[str, Any] = Field(..., description="最適化パラメータ")


class MultiObjectiveOptimizationRequest(BaseModel):
    """マルチ目的最適化リクエスト"""

    base_config: BacktestRequest = Field(..., description="基本設定")
    objectives: List[str] = Field(..., description="最適化対象の指標リスト")
    weights: Optional[List[float]] = Field(None, description="各指標の重み")
    optimization_params: Optional[Dict[str, Any]] = Field(
        None, description="追加の最適化パラメータ"
    )


class RobustnessTestRequest(BaseModel):
    """ロバストネステストリクエスト"""

    base_config: BacktestRequest = Field(..., description="基本設定")
    test_periods: List[List[str]] = Field(..., description="テスト期間のリスト")
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

    return await handle_api_exception(_run)


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

    return await handle_api_exception(_get_results)


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

    return await handle_api_exception(_get_by_id)


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

    return await handle_api_exception(_delete_result)


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

    return await handle_api_exception(_get_strategies)


@router.post("/optimize", response_model=BacktestResponse)
async def optimize_strategy(
    request: OptimizationRequest, db: Session = Depends(get_db)
):
    """
    戦略を最適化

    Args:
        request: 最適化リクエスト
        db: データベースセッション

    Returns:
        最適化結果
    """

    async def _optimize():
        backtest_service = BacktestService()
        base_config = _create_base_config(request.base_config)
        result = backtest_service.optimize_strategy(
            base_config, request.optimization_params
        )
        saved_result = _save_backtest_result(result, db)
        return {"success": True, "result": saved_result}

    return await handle_api_exception(_optimize)


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
        enhanced_backtest_service = EnhancedBacktestService()
        base_config = _create_base_config(request.base_config)
        result = enhanced_backtest_service.optimize_strategy_enhanced(
            base_config, request.optimization_params
        )
        saved_result = _save_backtest_result(result, db)
        return {"success": True, "result": saved_result}

    return await handle_api_exception(_optimize_enhanced)


@router.post("/multi-objective-optimization", response_model=BacktestResponse)
async def multi_objective_optimization(
    request: MultiObjectiveOptimizationRequest, db: Session = Depends(get_db)
):
    """
    戦略を多目的最適化

    Args:
        request: 多目的最適化リクエスト
        db: データベースセッション

    Returns:
        最適化結果
    """

    async def _multi_objective_optimize():
        enhanced_backtest_service = EnhancedBacktestService()
        base_config = _create_base_config(request.base_config)
        result = enhanced_backtest_service.multi_objective_optimization(
            base_config,
            request.objectives,
            request.weights,
            request.optimization_params,
        )
        saved_result = _save_backtest_result(result, db)
        return {"success": True, "result": saved_result}

    return await handle_api_exception(_multi_objective_optimize)


@router.post("/robustness-test", response_model=BacktestResponse)
async def robustness_test(
    request: RobustnessTestRequest, db: Session = Depends(get_db)
):
    """
    戦略をロバストネステスト

    Args:
        request: ロバストネステストリクエスト
        db: データベースセッション

    Returns:
        テスト結果
    """

    async def _robustness_test():
        enhanced_backtest_service = EnhancedBacktestService()
        base_config = _create_base_config(request.base_config)
        result = enhanced_backtest_service.robustness_test(
            base_config, request.test_periods, request.optimization_params
        )
        saved_result = _save_backtest_result(result, db)
        return {"success": True, "result": saved_result}

    return await handle_api_exception(_robustness_test)


@router.get("/health")
async def health_check():
    """
    ヘルスチェックエンドポイント

    Returns:
        APIの状態
    """

    async def _check_health():
        return {"status": "ok"}

    return await handle_api_exception(_check_health)
