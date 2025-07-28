"""
ML管理API

フロントエンド用のML管理機能を提供するAPIエンドポイント
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any
import logging
from sqlalchemy.orm import Session


from app.services.ml.orchestration.ml_management_orchestration_service import (
    MLManagementOrchestrationService,
)
from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
from app.services.ml.config import ml_config
from app.utils.unified_error_handler import UnifiedErrorHandler
from app.api.dependencies import get_ml_management_orchestration_service

from app.services.backtest.backtest_data_service import BacktestDataService
from app.utils.api_utils import APIResponseHelper
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.connection import get_db


router = APIRouter(prefix="/api/ml", tags=["ml_management"])
logger = logging.getLogger(__name__)


@router.get("/models")
async def get_models(
    ml_service: MLManagementOrchestrationService = Depends(
        get_ml_management_orchestration_service
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
        get_ml_management_orchestration_service
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
        get_ml_management_orchestration_service
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
        get_ml_management_orchestration_service
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


@router.get("/automl-feature-analysis")
async def get_automl_feature_analysis(
    top_n: int = 20,
    ml_service: MLManagementOrchestrationService = Depends(
        get_ml_management_orchestration_service
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
        get_ml_management_orchestration_service
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
async def update_ml_config(config_data: Dict[str, Any]):
    """
    ML設定を更新

    Args:
        config_data: 更新する設定データ
    """

    async def _update_ml_config():
        # TODO: 設定の更新ロジックを実装
        # 現在は読み取り専用として扱う
        logger.info(f"ML設定更新要求: {config_data}")
        return APIResponseHelper.api_response(
            success=True, message="設定が更新されました（現在は読み取り専用）"
        )

    return await UnifiedErrorHandler.safe_execute_async(_update_ml_config)


@router.post("/config/reset")
async def reset_ml_config():
    """
    ML設定をデフォルト値にリセット
    """

    async def _reset_ml_config():
        # TODO: 設定のリセットロジックを実装
        logger.info("ML設定リセット要求")
        return {
            "message": "設定がデフォルト値にリセットされました（現在は読み取り専用）"
        }

    return await UnifiedErrorHandler.safe_execute_async(_reset_ml_config)


@router.post("/models/cleanup")
async def cleanup_old_models(
    ml_service: MLManagementOrchestrationService = Depends(
        get_ml_management_orchestration_service
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
