"""
ベイジアン最適化API

GAパラメータとMLハイパーパラメータのベイジアン最適化を提供します。
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from app.core.services.optimization import BayesianOptimizer
from app.core.utils.unified_error_handler import UnifiedErrorHandler

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


class OptimizationResponse(BaseModel):
    """最適化レスポンス"""

    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    message: str
    timestamp: str


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

    return await UnifiedErrorHandler.safe_execute_async(_get_parameter_space)
