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
    try:
        backtest_service = BacktestService()
        config = _create_base_config(request)
        result = backtest_service.run_backtest(config)
        saved_result = _save_backtest_result(result, db)
        return BacktestResponse(success=True, result=saved_result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


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
    try:
        backtest_repo = BacktestResultRepository(db)

        results = backtest_repo.get_backtest_results(
            limit=limit, offset=offset, symbol=symbol, strategy_name=strategy_name
        )

        total = backtest_repo.count_backtest_results(
            symbol=symbol, strategy_name=strategy_name
        )

        return BacktestResultsResponse(success=True, results=results, total=total)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


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
    try:
        backtest_repo = BacktestResultRepository(db)
        result = backtest_repo.get_backtest_result_by_id(result_id)

        if result is None:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        return BacktestResponse(success=True, result=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


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
    try:
        backtest_repo = BacktestResultRepository(db)
        success = backtest_repo.delete_backtest_result(result_id)

        if not success:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        return {"success": True, "message": "Backtest result deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/strategies")
async def get_supported_strategies():
    """
    サポートされている戦略一覧を取得

    Returns:
        戦略一覧
    """
    try:
        backtest_service = BacktestService()
        strategies = backtest_service.get_supported_strategies()

        return {"success": True, "strategies": strategies}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/optimize", response_model=BacktestResponse)
async def optimize_strategy(
    request: OptimizationRequest, db: Session = Depends(get_db)
):
    """
    戦略パラメータを最適化

    Args:
        request: 最適化リクエスト
        db: データベースセッション

    Returns:
        最適化結果
    """
    try:
        backtest_service = BacktestService()
        base_config = _create_base_config(request.base_config)
        result = backtest_service.optimize_strategy(
            base_config, request.optimization_params
        )
        saved_result = _save_backtest_result(result, db)
        return BacktestResponse(success=True, result=saved_result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/optimize-enhanced", response_model=BacktestResponse)
async def optimize_strategy_enhanced(
    request: EnhancedOptimizationRequest, db: Session = Depends(get_db)
):
    """
    拡張戦略最適化を実行

    backtesting.py内蔵の高度な最適化機能（SAMBO、ヒートマップ、制約条件）を使用

    Args:
        request: 拡張最適化リクエスト
        db: データベースセッション

    Returns:
        最適化結果
    """
    try:
        enhanced_service = EnhancedBacktestService()
        base_config = _create_base_config(request.base_config)
        result = enhanced_service.optimize_strategy_enhanced(
            base_config, request.optimization_params
        )
        saved_result = _save_backtest_result(result, db)
        return BacktestResponse(success=True, result=saved_result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/multi-objective-optimization", response_model=BacktestResponse)
async def multi_objective_optimization(
    request: MultiObjectiveOptimizationRequest, db: Session = Depends(get_db)
):
    """
    マルチ目的最適化を実行

    複数の指標を同時に最適化

    Args:
        request: マルチ目的最適化リクエスト
        db: データベースセッション

    Returns:
        最適化結果
    """
    try:
        enhanced_service = EnhancedBacktestService()
        base_config = _create_base_config(request.base_config)
        result = enhanced_service.multi_objective_optimization(
            base_config,
            request.objectives,
            request.weights,
            request.optimization_params,
        )
        saved_result = _save_backtest_result(result, db)
        return BacktestResponse(success=True, result=saved_result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/robustness-test", response_model=BacktestResponse)
async def robustness_test(
    request: RobustnessTestRequest, db: Session = Depends(get_db)
):
    """
    ロバストネステストを実行

    複数期間での最適化を行い、戦略の安定性を評価

    Args:
        request: ロバストネステストリクエスト
        db: データベースセッション

    Returns:
        ロバストネステスト結果
    """
    try:
        enhanced_service = EnhancedBacktestService()
        base_config = _create_base_config(request.base_config)

        # `start_date`と`end_date`はロバストネステストでは使用しないため削除
        base_config.pop("start_date", None)
        base_config.pop("end_date", None)

        # テスト期間をタプルのリストに変換
        test_periods = [(period[0], period[1]) for period in request.test_periods]

        # ロバストネステストを実行
        result = enhanced_service.robustness_test(
            base_config, test_periods, request.optimization_params
        )

        # 結果をデータベースに保存（個別結果の最初のものを代表として保存）
        if result.get("individual_results"):
            first_result = next(
                (
                    res
                    for res in result["individual_results"].values()
                    if "error" not in res
                ),
                None,
            )
            if first_result:
                # ロバストネス情報を追加
                first_result["robustness_analysis"] = result.get("robustness_analysis")
                saved_result = _save_backtest_result(first_result, db)
                result["saved_result_id"] = saved_result.get("id")

        return BacktestResponse(success=True, result=result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/health")
async def health_check():
    """
    ヘルスチェック

    Returns:
        サービス状態
    """
    return {
        "success": True,
        "service": "backtest",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }
