"""
ベイジアン最適化API

GAパラメータとMLハイパーパラメータのベイジアン最適化を提供します。
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from datetime import datetime
import numpy as np

from database.connection import get_db, SessionLocal
from app.core.services.optimization import BayesianOptimizer
from app.core.services.backtest_service import BacktestService
from app.core.services.backtest_data_service import BacktestDataService
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.bayesian_optimization_repository import BayesianOptimizationRepository
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


class MLOptimizationRequest(BaseModel):
    """MLハイパーパラメータ最適化リクエスト"""
    model_type: str = Field(..., description="モデルタイプ")
    parameter_space: Optional[Dict[str, ParameterSpace]] = Field(None, description="パラメータ空間")
    n_calls: int = Field(30, description="最適化試行回数")
    optimization_config: Optional[Dict[str, Any]] = Field(None, description="最適化設定")

    # プロファイル保存用パラメータ
    save_as_profile: bool = Field(default=False, description="プロファイルとして保存するか")
    profile_name: Optional[str] = Field(None, description="プロファイル名")
    profile_description: Optional[str] = Field(None, description="プロファイル説明")


class ProfileUpdateRequest(BaseModel):
    """プロファイル更新リクエスト"""
    profile_name: Optional[str] = Field(None, description="プロファイル名")
    description: Optional[str] = Field(None, description="説明")
    is_default: Optional[bool] = Field(None, description="デフォルトプロファイルかどうか")
    is_active: Optional[bool] = Field(None, description="アクティブフラグ")
    target_model_type: Optional[str] = Field(None, description="対象モデルタイプ")


class OptimizationResponse(BaseModel):
    """最適化レスポンス"""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    message: str
    timestamp: str


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
        
        # NumPy型をPythonの標準型に変換する関数
        def convert_numpy_types(obj):
            """NumPy型をPythonの標準型に再帰的に変換"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        # 結果を整理（NumPy型を変換）
        result = {
            "model_type": request.model_type,
            "optimization_type": "bayesian_ml",
            "best_params": convert_numpy_types(optimization_result.best_params),
            "best_score": convert_numpy_types(optimization_result.best_score),
            "total_evaluations": convert_numpy_types(optimization_result.total_evaluations),
            "optimization_time": convert_numpy_types(optimization_result.optimization_time),
            "convergence_info": convert_numpy_types(optimization_result.convergence_info),
            "optimization_history": convert_numpy_types(optimization_result.optimization_history)
        }

        logger.info(f"MLハイパーパラメータ最適化完了: ベストスコア={optimization_result.best_score:.4f}")

        # プロファイルとして保存する場合
        if request.save_as_profile and request.profile_name:
            try:
                bayesian_repo = BayesianOptimizationRepository(db)
                saved_result = bayesian_repo.create_optimization_result(
                    profile_name=request.profile_name,
                    optimization_type="bayesian_ml",
                    model_type=request.model_type,
                    best_params=result["best_params"],
                    best_score=result["best_score"],
                    total_evaluations=result["total_evaluations"],
                    optimization_time=result["optimization_time"],
                    convergence_info=result["convergence_info"],
                    optimization_history=result["optimization_history"],
                    description=request.profile_description,
                    target_model_type=request.model_type,
                )

                result["saved_profile_id"] = saved_result.id
                logger.info(f"最適化結果をプロファイルとして保存: {request.profile_name}")

            except Exception as e:
                logger.warning(f"プロファイル保存エラー: {e}")
                # エラーが発生しても最適化結果は返す

        return {
            "success": True,
            "result": result,
            "message": "MLハイパーパラメータのベイジアン最適化が完了しました",
            "timestamp": datetime.now().isoformat()
        }
    
    return await APIErrorHandler.handle_api_exception(_optimize_ml)


# プロファイル管理エンドポイント（統合版）


@router.get("/profiles")
async def get_profiles(
    target_model_type: Optional[str] = None,
    include_inactive: bool = False,
    limit: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    最適化プロファイル一覧を取得
    """
    async def _get_profiles():
        bayesian_repo = BayesianOptimizationRepository(db)

        if target_model_type:
            profiles = bayesian_repo.get_profiles_by_model_type(
                target_model_type=target_model_type,
                include_inactive=include_inactive,
                limit=limit
            )
        else:
            profiles = bayesian_repo.get_all_results(
                include_inactive=include_inactive,
                limit=limit
            )

        return {
            "success": True,
            "profiles": [profile.to_dict() for profile in profiles],
            "count": len(profiles),
            "message": "プロファイル一覧を取得しました",
            "timestamp": datetime.now().isoformat()
        }

    return await APIErrorHandler.handle_api_exception(_get_profiles)


@router.get("/profiles/{profile_id}")
async def get_profile(
    profile_id: int,
    db: Session = Depends(get_db)
):
    """
    特定の最適化プロファイルを取得
    """
    async def _get_profile():
        bayesian_repo = BayesianOptimizationRepository(db)
        profile = bayesian_repo.get_by_id(profile_id)

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"プロファイルが見つかりません: ID={profile_id}"
            )

        return {
            "success": True,
            "profile": profile.to_dict(),
            "message": "プロファイルを取得しました",
            "timestamp": datetime.now().isoformat()
        }

    return await APIErrorHandler.handle_api_exception(_get_profile)


@router.put("/profiles/{profile_id}")
async def update_profile(
    profile_id: int,
    request: ProfileUpdateRequest,
    db: Session = Depends(get_db)
):
    """
    最適化プロファイルを更新
    """
    async def _update_profile():
        bayesian_repo = BayesianOptimizationRepository(db)

        # 更新データを準備
        update_data = {}
        if request.profile_name is not None:
            update_data["profile_name"] = request.profile_name
        if request.description is not None:
            update_data["description"] = request.description
        if request.is_default is not None:
            update_data["is_default"] = request.is_default
        if request.is_active is not None:
            update_data["is_active"] = request.is_active
        if request.target_model_type is not None:
            update_data["target_model_type"] = request.target_model_type

        profile = bayesian_repo.update_optimization_result(profile_id, **update_data)

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"プロファイルが見つかりません: ID={profile_id}"
            )

        return {
            "success": True,
            "profile": profile.to_dict(),
            "message": f"プロファイル '{profile.name}' を更新しました",
            "timestamp": datetime.now().isoformat()
        }

    return await APIErrorHandler.handle_api_exception(_update_profile)


@router.delete("/profiles/{profile_id}")
async def delete_profile(
    profile_id: int,
    db: Session = Depends(get_db)
):
    """
    最適化プロファイルを削除
    """
    async def _delete_profile():
        bayesian_repo = BayesianOptimizationRepository(db)
        success = bayesian_repo.delete_result(profile_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"プロファイルが見つかりません: ID={profile_id}"
            )

        return {
            "success": True,
            "message": f"プロファイル ID={profile_id} を削除しました",
            "timestamp": datetime.now().isoformat()
        }

    return await APIErrorHandler.handle_api_exception(_delete_profile)


@router.get("/profiles/default/{model_type}")
async def get_default_profile(
    model_type: str,
    db: Session = Depends(get_db)
):
    """
    指定されたモデルタイプのデフォルトプロファイルを取得
    """
    async def _get_default_profile():
        bayesian_repo = BayesianOptimizationRepository(db)
        profile = bayesian_repo.get_default_profile(target_model_type=model_type)

        if not profile:
            return {
                "success": True,
                "profile": None,
                "message": f"モデルタイプ '{model_type}' のデフォルトプロファイルが見つかりません",
                "timestamp": datetime.now().isoformat()
            }

        return {
            "success": True,
            "profile": profile.to_dict(),
            "message": f"モデルタイプ '{model_type}' のデフォルトプロファイルを取得しました",
            "timestamp": datetime.now().isoformat()
        }

    return await APIErrorHandler.handle_api_exception(_get_default_profile)


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
        
        if optimization_type == "lightgbm":
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
