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

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


# Pydanticモデル定義
class StrategyConfig(BaseModel):
    """戦略設定"""
    strategy_type: str = Field(..., description="戦略タイプ")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="戦略パラメータ")


class BacktestRequest(BaseModel):
    """バックテストリクエスト"""
    strategy_name: str = Field(..., description="戦略名")
    symbol: str = Field(..., description="取引ペア")
    timeframe: str = Field(..., description="時間軸")
    start_date: datetime = Field(..., description="開始日時")
    end_date: datetime = Field(..., description="終了日時")
    initial_capital: float = Field(..., gt=0, description="初期資金")
    commission_rate: float = Field(default=0.001, ge=0, le=1, description="手数料率")
    strategy_config: StrategyConfig = Field(..., description="戦略設定")


class OptimizationRequest(BaseModel):
    """最適化リクエスト"""
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
        # バックテストサービスを初期化
        backtest_service = BacktestService()
        
        # リクエストを辞書に変換
        config = {
            "strategy_name": request.strategy_name,
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "initial_capital": request.initial_capital,
            "commission_rate": request.commission_rate,
            "strategy_config": {
                "strategy_type": request.strategy_config.strategy_type,
                "parameters": request.strategy_config.parameters
            }
        }
        
        # バックテストを実行
        result = backtest_service.run_backtest(config)
        
        # 結果をデータベースに保存
        backtest_repo = BacktestResultRepository(db)
        saved_result = backtest_repo.save_backtest_result(result)
        
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
    db: Session = Depends(get_db)
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
            limit=limit,
            offset=offset,
            symbol=symbol,
            strategy_name=strategy_name
        )
        
        total = backtest_repo.count_backtest_results(
            symbol=symbol,
            strategy_name=strategy_name
        )
        
        return BacktestResultsResponse(
            success=True,
            results=results,
            total=total
        )
        
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
async def optimize_strategy(request: OptimizationRequest, db: Session = Depends(get_db)):
    """
    戦略パラメータを最適化
    
    Args:
        request: 最適化リクエスト
        db: データベースセッション
        
    Returns:
        最適化結果
    """
    try:
        # バックテストサービスを初期化
        backtest_service = BacktestService()
        
        # 基本設定を辞書に変換
        base_config = {
            "strategy_name": request.base_config.strategy_name,
            "symbol": request.base_config.symbol,
            "timeframe": request.base_config.timeframe,
            "start_date": request.base_config.start_date,
            "end_date": request.base_config.end_date,
            "initial_capital": request.base_config.initial_capital,
            "commission_rate": request.base_config.commission_rate,
            "strategy_config": {
                "strategy_type": request.base_config.strategy_config.strategy_type,
                "parameters": request.base_config.strategy_config.parameters
            }
        }
        
        # 最適化を実行
        result = backtest_service.optimize_strategy(base_config, request.optimization_params)
        
        # 結果をデータベースに保存
        backtest_repo = BacktestResultRepository(db)
        saved_result = backtest_repo.save_backtest_result(result)
        
        return BacktestResponse(success=True, result=saved_result)
        
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
        "timestamp": datetime.now().isoformat()
    }
