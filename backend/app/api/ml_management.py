"""
ML管理API

フロントエンド用のML管理機能を提供するAPIエンドポイント
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends

from app.api.dependencies import get_ml_management_orchestration_service
from app.services.ml.orchestration.ml_management_orchestration_service import (
    MLManagementOrchestrationService,
)
from app.utils.error_handler import ErrorHandler
from app.utils.response import result_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml", tags=["ml_management"])


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
        """学習済みモデルの一覧を取得するためのメインロジックを実行します。"""
        return await ml_service.get_formatted_models()

    return await ErrorHandler.safe_execute_async(_get_models)


@router.delete("/models/all")
async def delete_all_models(
    ml_service: MLManagementOrchestrationService = Depends(
        get_ml_management_orchestration_service
    ),
):
    """
    すべてのモデルを削除

    Args:
        ml_service: ML管理サービス（依存性注入）
    """

    async def _delete_all_models():
        """すべてのモデルを削除するためのメインロジックを実行します。"""
        return await ml_service.delete_all_models()

    return await ErrorHandler.safe_execute_async(_delete_all_models)


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
    logger.info(
        f"🗑️ 個別モデル削除エンドポイントが呼び出されました: model_id={model_id}"
    )

    async def _delete_model():
        """指定されたモデルを削除するためのメインロジックを実行します。"""
        return await ml_service.delete_model(model_id)

    return await ErrorHandler.safe_execute_async(_delete_model)


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
        """MLモデルの現在の状態を取得するためのメインロジックを実行します。"""
        return await ml_service.get_ml_status()

    return await ErrorHandler.safe_execute_async(_get_ml_status)


@router.get("/feature-importance")
async def get_feature_importance(
    top_n: int = 10,
    ml_service: MLManagementOrchestrationService = Depends(
        get_ml_management_orchestration_service
    ),
):
    """
    特徴量重要度を取得

    現在読み込まれているMLモデルの特徴量重要度を取得します。
    予測に最も影響を与えている特徴量を上位N件で返します。

    Args:
        top_n: 上位N件の特徴量を取得（デフォルト: 10）
        ml_service: ML管理サービス（依存性注入）

    Returns:
        特徴量名と重要度スコアのリスト

    Raises:
        HTTPException: モデルが読み込まれていない場合
    """

    async def _get_feature_importance():
        """特徴量重要度を取得するためのメインロジックを実行します。"""
        return await ml_service.get_feature_importance(top_n)

    return await ErrorHandler.safe_execute_async(_get_feature_importance)


@router.post("/models/{model_name}/load")
async def load_model(
    model_name: str,
    ml_service: MLManagementOrchestrationService = Depends(
        get_ml_management_orchestration_service
    ),
):
    """
    指定されたモデルを読み込み

    指定されたモデルファイルをメモリに読み込み、予測に使用できる状態にします。
    既に読み込まれているモデルがある場合は置き換えられます。

    Args:
        model_name: 読み込むモデル名（ファイル名）
        ml_service: ML管理サービス（依存性注入）

    Returns:
        読み込み結果を含むJSONレスポンス

    Raises:
        HTTPException: モデルファイルが存在しない場合や読み込みに失敗した場合
    """

    async def _load_model():
        """指定されたモデルを読み込むためのメインロジックを実行します。"""
        return await ml_service.load_model(model_name)

    return await ErrorHandler.safe_execute_async(_load_model)


@router.get("/models/current")
async def get_current_model(
    ml_service: MLManagementOrchestrationService = Depends(
        get_ml_management_orchestration_service
    ),
):
    """
    現在読み込まれているモデル情報を取得

    現在メモリに読み込まれているモデルの詳細情報（モデル名、
    トレーニング日時、パフォーマンス指標、ハイパーパラメータなど）を返します。

    Args:
        ml_service: ML管理サービス（依存性注入）

    Returns:
        現在のモデル情報を含むJSONレスポンス

    Raises:
        HTTPException: モデルが読み込まれていない場合
    """

    async def _get_current_model():
        """現在読み込まれているモデル情報を取得するためのメインロジックを実行します。"""
        return await ml_service.get_current_model_info()

    return await ErrorHandler.safe_execute_async(_get_current_model)


# AutoML機能は削除されたため、/automl-feature-analysisエンドポイントは削除されました
# 特徴量分析は /feature-importance エンドポイントを使用してください


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
        """ML設定を取得するためのメインロジックを実行します。"""
        return ml_service.get_ml_config_dict()

    return await ErrorHandler.safe_execute_async(_get_ml_config)


@router.put("/config")
async def update_ml_config(
    config_data: Dict[str, Any],
    ml_service: MLManagementOrchestrationService = Depends(
        get_ml_management_orchestration_service
    ),
):
    """
    ML設定を更新

    実行時のML関連設定を動的に更新します。
    変更は即時に反映され、次回のトレーニングから適用されます。

    Args:
        config_data: 更新する設定データ（キーと値のペア）
        ml_service: ML管理サービス（依存性注入）

    Returns:
        更新結果と更新後の設定を含むJSONレスポンス

    Raises:
        HTTPException: 設定キーが無効な場合や値の形式が不正な場合
    """

    async def _update_ml_config():
        """ML設定を更新するためのメインロジックを実行します。"""
        logger.info(f"ML設定更新要求: {config_data}")
        result = await ml_service.update_ml_config(config_data)

        return result_response(
            success=result["success"],
            message=result["message"],
            data=result.get("updated_config"),
        )

    return await ErrorHandler.safe_execute_async(_update_ml_config)


@router.post("/config/reset")
async def reset_ml_config(
    ml_service: MLManagementOrchestrationService = Depends(
        get_ml_management_orchestration_service
    ),
):
    """
    ML設定をデフォルト値にリセット

    すべてのML関連設定をシステムデフォルト値に戻します。

    Args:
        ml_service: ML管理サービス（依存性注入）

    Returns:
        リセット結果とデフォルト設定を含むJSONレスポンス
    """

    async def _reset_ml_config():
        """ML設定をデフォルト値にリセットするためのメインロジックを実行します。"""
        logger.info("ML設定リセット要求")
        result = await ml_service.reset_ml_config()

        return result_response(
            success=result["success"],
            message=result["message"],
            data=result.get("config"),
        )

    return await ErrorHandler.safe_execute_async(_reset_ml_config)


@router.post("/models/cleanup")
async def cleanup_old_models(
    ml_service: MLManagementOrchestrationService = Depends(
        get_ml_management_orchestration_service
    ),
):
    """
    古いモデルファイルをクリーンアップ

    ディスクに保存されている古いモデルファイルを自動的に削除します。
    最新のN件のモデルは保持され、それ以前のモデルが削除されます。

    Args:
        ml_service: ML管理サービス（依存性注入）

    Returns:
        削除されたモデル数とクリーンアップ結果を含むJSONレスポンス
    """

    async def _cleanup_old_models():
        """古いモデルファイルをクリーンアップするためのメインロジックを実行します。"""
        return await ml_service.cleanup_old_models()

    return await ErrorHandler.safe_execute_async(_cleanup_old_models)
