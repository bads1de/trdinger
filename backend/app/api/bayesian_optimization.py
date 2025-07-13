"""
ベイジアン最適化API

GAパラメータとMLハイパーパラメータのベイジアン最適化を提供します。
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database.connection import get_db, SessionLocal
from app.core.services.optimization import BayesianOptimizer
from app.core.services.backtest_service import BacktestService
from app.core.services.backtest_data_service import BacktestDataService
from database.repositories.ohlcv_repository import OHLCVRepository
from app.core.utils.api_utils import APIErrorHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/bayesian-optimization", tags=["bayesian-optimization"])


# リクエスト/レスポンスモデル
class ParameterSpace(BaseModel):
    """パラメータ空間の定義"""
    type: str = Field(..., description="パラメータタイプ (real, integer, categorical)")
    low: Optional[float] = Field(None, description="最小値 (real, integer)")
    high: Optional[float] = Field(None, description="最大値 (real, integer)")
    categories: Optional[List[Any]] = Field(None, description="カテゴリ値 (categorical)")


class GAOptimizationRequest(BaseModel):
    """GAパラメータ最適化リクエスト"""
    experiment_name: str = Field(..., description="実験名")
    base_config: Dict[str, Any] = Field(..., description="基本バックテスト設定")
    parameter_space: Optional[Dict[str, ParameterSpace]] = Field(None, description="パラメータ空間")
    n_calls: int = Field(50, description="最適化試行回数")
    optimization_config: Optional[Dict[str, Any]] = Field(None, description="最適化設定")


class MLOptimizationRequest(BaseModel):
    """MLハイパーパラメータ最適化リクエスト"""
    model_type: str = Field(..., description="モデルタイプ")
    parameter_space: Optional[Dict[str, ParameterSpace]] = Field(None, description="パラメータ空間")
    n_calls: int = Field(30, description="最適化試行回数")
    optimization_config: Optional[Dict[str, Any]] = Field(None, description="最適化設定")


class OptimizationResponse(BaseModel):
    """最適化レスポンス"""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    message: str
    timestamp: str


@router.post("/ga-parameters", response_model=OptimizationResponse)
async def optimize_ga_parameters(
    request: GAOptimizationRequest,
    db: Session = Depends(get_db)
):
    """
    GAパラメータのベイジアン最適化
    
    Args:
        request: GAパラメータ最適化リクエスト
        db: データベースセッション
    
    Returns:
        最適化結果
    """
    
    async def _optimize_ga():
        logger.info(f"GAパラメータのベイジアン最適化を開始: {request.experiment_name}")
        
        # ベイジアン最適化エンジンを初期化
        optimizer = BayesianOptimizer()
        
        # バックテストサービスを初期化
        backtest_service = BacktestService()
        ohlcv_repo = OHLCVRepository(db)
        data_service = BacktestDataService(ohlcv_repo)
        
        # 目的関数を定義（バックテストを実行してスコアを返す）
        def objective_function(params: Dict[str, Any]) -> float:
            try:
                # GAパラメータを基本設定に適用
                config = request.base_config.copy()
                
                # GAパラメータを戦略設定に反映
                if "strategy_config" not in config:
                    config["strategy_config"] = {}
                
                # GAパラメータを戦略設定に追加
                config["strategy_config"].update(params)
                
                # バックテストを実行
                result = backtest_service.run_backtest(config)
                
                # スコアを抽出（SQNを使用）
                if "stats" in result and "SQN" in result["stats"]:
                    return float(result["stats"]["SQN"])
                else:
                    return 0.0
                    
            except Exception as e:
                logger.warning(f"目的関数評価エラー: {e}")
                return -1000.0  # ペナルティスコア
        
        # パラメータ空間を変換
        parameter_space_dict = None
        if request.parameter_space:
            parameter_space_dict = {}
            for param_name, param_config in request.parameter_space.items():
                parameter_space_dict[param_name] = {
                    "type": param_config.type,
                    "low": param_config.low,
                    "high": param_config.high,
                    "categories": param_config.categories
                }
        
        # ベイジアン最適化を実行
        optimization_result = optimizer.optimize_ga_parameters(
            objective_function=objective_function,
            parameter_space=parameter_space_dict,
            n_calls=request.n_calls
        )
        
        # 結果を整理
        result = {
            "experiment_name": request.experiment_name,
            "optimization_type": "bayesian_ga",
            "best_params": optimization_result.best_params,
            "best_score": optimization_result.best_score,
            "total_evaluations": optimization_result.total_evaluations,
            "optimization_time": optimization_result.optimization_time,
            "convergence_info": optimization_result.convergence_info,
            "optimization_history": optimization_result.optimization_history
        }
        
        logger.info(f"GAパラメータ最適化完了: ベストスコア={optimization_result.best_score:.4f}")
        
        return {
            "success": True,
            "result": result,
            "message": "GAパラメータのベイジアン最適化が完了しました"
        }
    
    return await APIErrorHandler.handle_api_exception(_optimize_ga)


@router.post("/ml-hyperparameters", response_model=OptimizationResponse)
async def optimize_ml_hyperparameters(
    request: MLOptimizationRequest,
    db: Session = Depends(get_db)
):
    """
    MLハイパーパラメータのベイジアン最適化
    
    Args:
        request: MLハイパーパラメータ最適化リクエスト
        db: データベースセッション
    
    Returns:
        最適化結果
    """
    
    async def _optimize_ml():
        logger.info(f"MLハイパーパラメータのベイジアン最適化を開始: {request.model_type}")
        
        # ベイジアン最適化エンジンを初期化
        optimizer = BayesianOptimizer()
        
        # 目的関数を定義（MLモデルの性能評価）
        def objective_function(params: Dict[str, Any]) -> float:
            try:
                # TODO: MLモデルの訓練と評価を実装
                # 現在はダミー実装
                logger.info(f"MLハイパーパラメータ評価: {params}")
                
                # ダミースコア（実際にはMLモデルの性能指標を返す）
                import random
                return random.uniform(0.5, 0.9)
                
            except Exception as e:
                logger.warning(f"ML目的関数評価エラー: {e}")
                return 0.0
        
        # パラメータ空間を変換
        parameter_space_dict = None
        if request.parameter_space:
            parameter_space_dict = {}
            for param_name, param_config in request.parameter_space.items():
                parameter_space_dict[param_name] = {
                    "type": param_config.type,
                    "low": param_config.low,
                    "high": param_config.high,
                    "categories": param_config.categories
                }
        
        # ベイジアン最適化を実行
        optimization_result = optimizer.optimize_ml_hyperparameters(
            model_type=request.model_type,
            objective_function=objective_function,
            parameter_space=parameter_space_dict,
            n_calls=request.n_calls
        )
        
        # 結果を整理
        result = {
            "model_type": request.model_type,
            "optimization_type": "bayesian_ml",
            "best_params": optimization_result.best_params,
            "best_score": optimization_result.best_score,
            "total_evaluations": optimization_result.total_evaluations,
            "optimization_time": optimization_result.optimization_time,
            "convergence_info": optimization_result.convergence_info,
            "optimization_history": optimization_result.optimization_history
        }
        
        logger.info(f"MLハイパーパラメータ最適化完了: ベストスコア={optimization_result.best_score:.4f}")
        
        return {
            "success": True,
            "result": result,
            "message": "MLハイパーパラメータのベイジアン最適化が完了しました"
        }
    
    return await APIErrorHandler.handle_api_exception(_optimize_ml)


@router.get("/parameter-spaces/{optimization_type}")
async def get_default_parameter_space(optimization_type: str):
    """
    デフォルトパラメータ空間を取得
    
    Args:
        optimization_type: 最適化タイプ (ga, lightgbm, etc.)
    
    Returns:
        デフォルトパラメータ空間
    """
    
    async def _get_parameter_space():
        optimizer = BayesianOptimizer()
        
        if optimization_type == "ga":
            parameter_space = optimizer._get_default_ga_parameter_space()
        elif optimization_type == "lightgbm":
            parameter_space = optimizer._get_default_ml_parameter_space("lightgbm")
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported optimization type: {optimization_type}"
            )
        
        return {
            "success": True,
            "parameter_space": parameter_space,
            "message": f"デフォルトパラメータ空間を取得しました: {optimization_type}"
        }
    
    return await APIErrorHandler.handle_api_exception(_get_parameter_space)
