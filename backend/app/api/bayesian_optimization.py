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

from database.connection import get_db
from app.core.services.optimization import BayesianOptimizer
from app.core.dependencies import get_bayesian_optimizer_with_db
from database.repositories.bayesian_optimization_repository import (
    BayesianOptimizationRepository,
)
from app.core.utils.api_utils import APIErrorHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/bayesian-optimization", tags=["bayesian-optimization"])


# リクエスト/レスポンスモデル
class ParameterSpace(BaseModel):
    """パラメータ空間の定義"""

    type: str = Field(..., description="パラメータタイプ (real, integer, categorical)")
    low: Optional[float] = Field(None, description="最小値 (real, integer)")
    high: Optional[float] = Field(None, description="最大値 (real, integer)")
    categories: Optional[List[Any]] = Field(
        None, description="カテゴリ値 (categorical)"
    )


class MLOptimizationRequest(BaseModel):
    """MLハイパーパラメータ最適化リクエスト"""

    model_type: str = Field(..., description="モデルタイプ")
    parameter_space: Optional[Dict[str, ParameterSpace]] = Field(
        None, description="パラメータ空間"
    )
    n_calls: int = Field(30, description="最適化試行回数")
    optimization_config: Optional[Dict[str, Any]] = Field(
        None, description="最適化設定"
    )

    # プロファイル保存用パラメータ
    save_as_profile: bool = Field(
        default=False, description="プロファイルとして保存するか"
    )
    profile_name: Optional[str] = Field(None, description="プロファイル名")
    profile_description: Optional[str] = Field(None, description="プロファイル説明")


class ProfileUpdateRequest(BaseModel):
    """プロファイル更新リクエスト"""

    profile_name: Optional[str] = Field(None, description="プロファイル名")
    description: Optional[str] = Field(None, description="説明")
    is_default: Optional[bool] = Field(
        None, description="デフォルトプロファイルかどうか"
    )
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
    request: MLOptimizationRequest, db: Session = Depends(get_db)
):
    """
    MLハイパーパラメータのベイジアン最適化

    Args:
        request: MLハイパーパラメータ最適化リクエスト
        db: データベースセッション

    Returns:
        最適化結果
    """
    # ビジネスロジックをサービス層に委譲
    optimizer = get_bayesian_optimizer_with_db(db)

    def execute_optimization():
        result = optimizer.execute_ml_optimization(
            model_type=request.model_type,
            parameter_space=request.parameter_space,
            n_calls=request.n_calls,
            save_as_profile=request.save_as_profile,
            profile_name=request.profile_name,
            profile_description=request.profile_description,
            db_session=db,
        )

        return {
            "success": True,
            "result": {
                "model_type": request.model_type,
                "optimization_type": "bayesian_ml",
                **result,
            },
            "message": "MLハイパーパラメータのベイジアン最適化が完了しました",
            "timestamp": datetime.now().isoformat(),
        }

    return await APIErrorHandler.handle_api_exception(execute_optimization)


# プロファイル管理エンドポイント（統合版）


@router.get("/profiles")
async def get_profiles(
    target_model_type: Optional[str] = None,
    include_inactive: bool = False,
    limit: Optional[int] = None,
    db: Session = Depends(get_db),
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
                limit=limit,
            )
        else:
            profiles = bayesian_repo.get_all_results(
                include_inactive=include_inactive, limit=limit
            )

        return {
            "success": True,
            "profiles": [profile.to_dict() for profile in profiles],
            "count": len(profiles),
            "message": "プロファイル一覧を取得しました",
            "timestamp": datetime.now().isoformat(),
        }

    return await APIErrorHandler.handle_api_exception(_get_profiles)


@router.get("/profiles/{profile_id}")
async def get_profile(profile_id: int, db: Session = Depends(get_db)):
    """
    特定の最適化プロファイルを取得
    """

    async def _get_profile():
        bayesian_repo = BayesianOptimizationRepository(db)
        profile = bayesian_repo.get_by_id(profile_id)

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"プロファイルが見つかりません: ID={profile_id}",
            )

        return {
            "success": True,
            "profile": profile.to_dict(),
            "message": "プロファイルを取得しました",
            "timestamp": datetime.now().isoformat(),
        }

    return await APIErrorHandler.handle_api_exception(_get_profile)


@router.put("/profiles/{profile_id}")
async def update_profile(
    profile_id: int, request: ProfileUpdateRequest, db: Session = Depends(get_db)
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
                detail=f"プロファイルが見つかりません: ID={profile_id}",
            )

        return {
            "success": True,
            "profile": profile.to_dict(),
            "message": f"プロファイル '{profile.name}' を更新しました",
            "timestamp": datetime.now().isoformat(),
        }

    return await APIErrorHandler.handle_api_exception(_update_profile)


@router.delete("/profiles/{profile_id}")
async def delete_profile(profile_id: int, db: Session = Depends(get_db)):
    """
    最適化プロファイルを削除
    """

    async def _delete_profile():
        bayesian_repo = BayesianOptimizationRepository(db)
        success = bayesian_repo.delete_result(profile_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"プロファイルが見つかりません: ID={profile_id}",
            )

        return {
            "success": True,
            "message": f"プロファイル ID={profile_id} を削除しました",
            "timestamp": datetime.now().isoformat(),
        }

    return await APIErrorHandler.handle_api_exception(_delete_profile)


@router.get("/profiles/default/{model_type}")
async def get_default_profile(model_type: str, db: Session = Depends(get_db)):
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
                "timestamp": datetime.now().isoformat(),
            }

        return {
            "success": True,
            "profile": profile.to_dict(),
            "message": f"モデルタイプ '{model_type}' のデフォルトプロファイルを取得しました",
            "timestamp": datetime.now().isoformat(),
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
                detail=f"Unsupported optimization type: {optimization_type}",
            )

        return {
            "success": True,
            "parameter_space": parameter_space,
            "message": f"デフォルトパラメータ空間を取得しました: {optimization_type}",
        }

    return await APIErrorHandler.handle_api_exception(_get_parameter_space)
