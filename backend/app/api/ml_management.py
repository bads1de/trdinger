"""
ML管理API

フロントエンド用のML管理機能を提供するAPIエンドポイント
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.services.backtest.backtest_data_service import BacktestDataService
from app.services.ml.orchestration.ml_management_orchestration_service import (
    MLManagementOrchestrationService,
)
from app.utils.api_utils import APIResponseHelper
from app.utils.unified_error_handler import UnifiedErrorHandler
from database.connection import get_db
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml", tags=["ml_management"])


@router.get("/models")
async def get_models(
    ml_service: MLManagementOrchestrationService = Depends(
        MLManagementOrchestrationService
    ),
):
    """
    学習済みモデルの一覧を取得

    Args:
        ml_service: ML管理サービス（依存性注入）

    Returns:
        モデル一覧
    """

    async def _get_models():
        return await ml_service.get_formatted_models()

    return await UnifiedErrorHandler.safe_execute_async(_get_models)


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    ml_service: MLManagementOrchestrationService = Depends(
        MLManagementOrchestrationService
    ),
):
    """
    指定されたモデルを削除

    Args:
        model_id: モデルID（ファイル名）
        ml_service: ML管理サービス（依存性注入）
    """

    async def _delete_model():
        return await ml_service.delete_model(model_id)

    return await UnifiedErrorHandler.safe_execute_async(_delete_model)


@router.get("/status")
async def get_ml_status(
    ml_service: MLManagementOrchestrationService = Depends(
        MLManagementOrchestrationService
    ),
):
    """
    MLモデルの現在の状態を取得

    Args:
        ml_service: ML管理サービス（依存性注入）

    Returns:
        モデル状態情報
    """

    async def _get_ml_status():
        return await ml_service.get_ml_status()

    return await UnifiedErrorHandler.safe_execute_async(_get_ml_status)


@router.get("/feature-importance")
async def get_feature_importance(
    top_n: int = 10,
    ml_service: MLManagementOrchestrationService = Depends(
        MLManagementOrchestrationService
    ),
):
    """
    特徴量重要度を取得

    Args:
        top_n: 上位N件の特徴量を取得
        ml_service: ML管理サービス（依存性注入）
    """

    async def _get_feature_importance():
        return await ml_service.get_feature_importance(top_n)

    return await UnifiedErrorHandler.safe_execute_async(_get_feature_importance)


@router.post("/models/{model_name}/load")
async def load_model(
    model_name: str,
    ml_service: MLManagementOrchestrationService = Depends(
        MLManagementOrchestrationService
    ),
):
    """
    指定されたモデルを読み込み

    Args:
        model_name: 読み込むモデル名
    """

    async def _load_model():
        return await ml_service.load_model(model_name)

    return await UnifiedErrorHandler.safe_execute_async(_load_model)


@router.get("/models/current")
async def get_current_model(
    ml_service: MLManagementOrchestrationService = Depends(
        MLManagementOrchestrationService
    ),
):
    """
    現在読み込まれているモデル情報を取得
    """

    async def _get_current_model():
        return await ml_service.get_current_model_info()

    return await UnifiedErrorHandler.safe_execute_async(_get_current_model)


@router.get("/automl-feature-analysis")
async def get_automl_feature_analysis(
    top_n: int = 20,
    ml_service: MLManagementOrchestrationService = Depends(
        MLManagementOrchestrationService
    ),
):
    """
    AutoML特徴量分析結果を取得

    Args:
        top_n: 上位N件の特徴量を取得
        ml_service: ML管理サービス（依存性注入）
    """

    async def _get_automl_feature_analysis():
        return await ml_service.get_automl_feature_analysis(top_n)

    return await UnifiedErrorHandler.safe_execute_async(_get_automl_feature_analysis)


@router.get("/config")
async def get_ml_config(
    ml_service: MLManagementOrchestrationService = Depends(
        MLManagementOrchestrationService
    ),
):
    """
    ML設定を取得

    Args:
        ml_service: ML管理サービス（依存性注入）

    Returns:
        ML設定
    """

    async def _get_ml_config():
        return ml_service.get_ml_config_dict()

    return await UnifiedErrorHandler.safe_execute_async(_get_ml_config)


@router.put("/config")
async def update_ml_config(
    config_data: Dict[str, Any],
    ml_service: MLManagementOrchestrationService = Depends(
        MLManagementOrchestrationService
    ),
):
    """
    ML設定を更新

    Args:
        config_data: 更新する設定データ
        ml_service: ML管理サービス（依存性注入）
    """

    async def _update_ml_config():
        logger.info(f"ML設定更新要求: {config_data}")
        result = await ml_service.update_ml_config(config_data)

        if result["success"]:
            return APIResponseHelper.api_response(
                success=True,
                message=result["message"],
                data=result.get("updated_config"),
            )
        else:
            return APIResponseHelper.api_response(
                success=False, message=result["message"]
            )

    return await UnifiedErrorHandler.safe_execute_async(_update_ml_config)


@router.post("/config/reset")
async def reset_ml_config(
    ml_service: MLManagementOrchestrationService = Depends(
        MLManagementOrchestrationService
    ),
):
    """
    ML設定をデフォルト値にリセット

    Args:
        ml_service: ML管理サービス（依存性注入）
    """

    async def _reset_ml_config():
        logger.info("ML設定リセット要求")
        result = await ml_service.reset_ml_config()

        if result["success"]:
            return APIResponseHelper.api_response(
                success=True, message=result["message"], data=result.get("config")
            )
        else:
            return APIResponseHelper.api_response(
                success=False, message=result["message"]
            )

    return await UnifiedErrorHandler.safe_execute_async(_reset_ml_config)


@router.post("/models/cleanup")
async def cleanup_old_models(
    ml_service: MLManagementOrchestrationService = Depends(
        MLManagementOrchestrationService
    ),
):
    """
    古いモデルファイルをクリーンアップ

    Args:
        ml_service: ML管理サービス（依存性注入）
    """

    async def _cleanup_old_models():
        return await ml_service.cleanup_old_models()

    return await UnifiedErrorHandler.safe_execute_async(_cleanup_old_models)


def get_data_service(db: Session = Depends(get_db)) -> BacktestDataService:
    """データサービスの依存性注入"""
    ohlcv_repo = OHLCVRepository(db)
    oi_repo = OpenInterestRepository(db)
    fr_repo = FundingRateRepository(db)
    return BacktestDataService(ohlcv_repo, oi_repo, fr_repo)
