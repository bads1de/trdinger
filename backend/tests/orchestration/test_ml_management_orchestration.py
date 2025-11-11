"""
ML管理オーケストレーションサービスのテストモジュール

MLManagementOrchestrationServiceの正常系、異常系、エッジケースをテストします。
"""

import os
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.services.ml.orchestration.ml_management_orchestration_service import (
    MLManagementOrchestrationService,
)


@pytest.fixture
def orchestration_service() -> MLManagementOrchestrationService:
    """
    MLManagementOrchestrationServiceのインスタンス

    Returns:
        MLManagementOrchestrationService: テスト対象のサービス
    """
    return MLManagementOrchestrationService()


@pytest.fixture
def sample_model_info() -> Dict[str, Any]:
    """
    サンプルモデル情報

    Returns:
        Dict[str, Any]: モデル情報のサンプルデータ
    """
    return {
        "name": "lightgbm_model_20240101_120000.pkl",
        "path": "/models/lightgbm_model_20240101_120000.pkl",
        "size_mb": 2.5,
        "modified_at": datetime(2024, 1, 1, 12, 0, 0),
        "directory": "/models",
    }


@pytest.fixture
def sample_model_metadata() -> Dict[str, Any]:
    """
    サンプルモデルメタデータ

    Returns:
        Dict[str, Any]: モデルメタデータのサンプルデータ
    """
    return {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "f1_score": 0.85,
        "auc_roc": 0.90,
        "model_type": "LightGBM",
        "feature_count": 50,
        "training_samples": 10000,
        "classification_report": {
            "macro avg": {
                "precision": 0.82,
                "recall": 0.88,
                "f1-score": 0.85,
            }
        },
    }


class TestServiceInitialization:
    """正常系: サービスの初期化テスト"""

    def test_service_creation(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        正常系: サービスが正常に初期化される

        Args:
            orchestration_service: オーケストレーションサービス
        """
        assert orchestration_service is not None
        assert isinstance(orchestration_service, MLManagementOrchestrationService)


class TestGetFormattedModels:
    """正常系: モデル一覧取得のテスト"""

    @pytest.mark.asyncio
    async def test_get_formatted_models_success(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_info: Dict[str, Any],
        sample_model_metadata: Dict[str, Any],
    ):
        """
        正常系: モデル一覧が正常に取得できる

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_info: サンプルモデル情報
            sample_model_metadata: サンプルメタデータ
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.list_models.return_value = [sample_model_info]
            mock_manager.load_model.return_value = {"metadata": sample_model_metadata}

            result = await orchestration_service.get_formatted_models()

            assert "models" in result
            assert len(result["models"]) == 1
            assert result["models"][0]["accuracy"] == 0.85

    @pytest.mark.asyncio
    async def test_get_formatted_models_empty(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        エッジケース: モデルが存在しない場合

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.list_models.return_value = []

            result = await orchestration_service.get_formatted_models()

            assert "models" in result
            assert len(result["models"]) == 0

    @pytest.mark.asyncio
    async def test_get_formatted_models_without_metadata(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_info: Dict[str, Any],
    ):
        """
        エッジケース: メタデータが存在しない場合

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_info: サンプルモデル情報
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.list_models.return_value = [sample_model_info]
            mock_manager.load_model.side_effect = Exception("Load error")

            result = await orchestration_service.get_formatted_models()

            assert "models" in result
            assert len(result["models"]) == 1
            # メタデータ読み込みエラー時はデフォルト値が設定される
            assert result["models"][0]["accuracy"] == 0.0
            assert result["models"][0]["model_type"] == "Unknown"
            assert result["models"][0]["training_samples"] == 0


class TestDeleteModel:
    """正常系: モデル削除のテスト"""

    @pytest.mark.asyncio
    async def test_delete_model_success(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_info: Dict[str, Any],
    ):
        """
        正常系: モデルが正常に削除される

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_info: サンプルモデル情報
        """
        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.os.path.exists"
            ) as mock_exists,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.os.remove"
            ) as mock_remove,
        ):
            mock_manager.list_models.return_value = [sample_model_info]
            mock_exists.return_value = True

            result = await orchestration_service.delete_model(
                model_id=sample_model_info["name"]
            )

            assert result["success"] is True
            mock_remove.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_model_not_found(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        異常系: 存在しないモデルの削除でHTTPExceptionが発生する

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.list_models.return_value = []

            with pytest.raises(HTTPException) as exc_info:
                await orchestration_service.delete_model(model_id="nonexistent")

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_model_file_not_exists(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_info: Dict[str, Any],
    ):
        """
        異常系: ファイルが存在しない場合

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_info: サンプルモデル情報
        """
        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.os.path.exists"
            ) as mock_exists,
        ):
            mock_manager.list_models.return_value = [sample_model_info]
            mock_exists.return_value = False

            with pytest.raises(HTTPException) as exc_info:
                await orchestration_service.delete_model(
                    model_id=sample_model_info["name"]
                )

            assert exc_info.value.status_code == 404


class TestDeleteAllModels:
    """正常系: 全モデル削除のテスト"""

    @pytest.mark.asyncio
    async def test_delete_all_models_success(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_info: Dict[str, Any],
    ):
        """
        正常系: 全モデルが正常に削除される

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_info: サンプルモデル情報
        """
        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.os.path.exists"
            ) as mock_exists,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.os.remove"
            ),
        ):
            mock_manager.list_models.return_value = [sample_model_info]
            mock_exists.return_value = True

            result = await orchestration_service.delete_all_models()

            assert result["success"] is True
            assert result["deleted_count"] == 1

    @pytest.mark.asyncio
    async def test_delete_all_models_empty(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        エッジケース: モデルが存在しない場合

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.list_models.return_value = []

            result = await orchestration_service.delete_all_models()

            assert result["success"] is True
            assert result["deleted_count"] == 0


class TestGetMLStatus:
    """正常系: MLステータス取得のテスト"""

    @pytest.mark.asyncio
    async def test_get_ml_status_with_model(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_metadata: Dict[str, Any],
    ):
        """
        正常系: モデルが存在する場合のステータス取得

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_metadata: サンプルメタデータ
        """
        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.os.path.exists"
            ) as mock_exists,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.os.path.getmtime"
            ) as mock_getmtime,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.os.path.getsize"
            ) as mock_getsize,
        ):
            mock_manager.get_latest_model.return_value = "/models/test.pkl"
            mock_exists.return_value = True
            mock_getmtime.return_value = 1640995200.0  # 2022-01-01 00:00:00
            mock_getsize.return_value = 1024 * 1024  # 1MB
            mock_manager.load_model.return_value = {"metadata": sample_model_metadata}

            result = await orchestration_service.get_ml_status()

            assert result["is_model_loaded"] is True
            assert result["is_trained"] is True

    @pytest.mark.asyncio
    async def test_get_ml_status_no_model(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        エッジケース: モデルが存在しない場合

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.get_latest_model.return_value = None

            result = await orchestration_service.get_ml_status()

            assert result["is_model_loaded"] is False
            assert result["status"] == "no_model"


class TestGetFeatureImportance:
    """正常系: 特徴量重要度取得のテスト"""

    @pytest.mark.asyncio
    async def test_get_feature_importance_success(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        正常系: 特徴量重要度が正常に取得できる

        Args:
            orchestration_service: オーケストレーションサービス
        """
        feature_importance = {
            "feature1": 0.5,
            "feature2": 0.3,
            "feature3": 0.2,
        }

        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.os.path.exists"
            ) as mock_exists,
        ):
            mock_manager.get_latest_model.return_value = "/models/test.pkl"
            mock_exists.return_value = True
            mock_manager.load_model.return_value = {
                "metadata": {"feature_importance": feature_importance}
            }

            result = await orchestration_service.get_feature_importance(top_n=3)

            assert "feature_importance" in result
            assert len(result["feature_importance"]) == 3

    @pytest.mark.asyncio
    async def test_get_feature_importance_no_model(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        エッジケース: モデルが存在しない場合

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.get_latest_model.return_value = None

            result = await orchestration_service.get_feature_importance()

            assert result["feature_importance"] == []


class TestMLConfigManagement:
    """正常系: ML設定管理のテスト"""

    def test_get_ml_config_dict(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        正常系: ML設定が正常に取得できる

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.ml_config_manager"
        ) as mock_config_manager:
            mock_config_manager.get_config_dict.return_value = {
                "model_type": "lightgbm",
                "learning_rate": 0.1,
            }

            result = orchestration_service.get_ml_config_dict()

            assert "model_type" in result
            assert result["model_type"] == "lightgbm"

    @pytest.mark.asyncio
    async def test_update_ml_config_success(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        正常系: ML設定が正常に更新される

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.ml_config_manager"
        ) as mock_config_manager:
            mock_config_manager.update_config.return_value = True
            mock_config_manager.get_config_dict.return_value = {"learning_rate": 0.05}

            result = await orchestration_service.update_ml_config(
                config_updates={"learning_rate": 0.05}
            )

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_reset_ml_config_success(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        正常系: ML設定が正常にリセットされる

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.ml_config_manager"
        ) as mock_config_manager:
            mock_config_manager.reset_config.return_value = True
            mock_config_manager.get_config_dict.return_value = {}

            result = await orchestration_service.reset_ml_config()

            assert result["success"] is True


class TestCleanupOldModels:
    """正常系: 古いモデルクリーンアップのテスト"""

    @pytest.mark.asyncio
    async def test_cleanup_old_models_success(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        正常系: 古いモデルが正常にクリーンアップされる

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.cleanup_expired_models.return_value = None

            result = await orchestration_service.cleanup_old_models()

            assert "message" in result
            mock_manager.cleanup_expired_models.assert_called_once()


class TestLoadModel:
    """正常系: モデル読み込みのテスト"""

    @pytest.mark.asyncio
    async def test_load_model_success(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_info: Dict[str, Any],
    ):
        """
        正常系: モデルが正常に読み込まれる

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_info: サンプルモデル情報
        """
        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch.object(
                orchestration_service, "get_current_model_info", new_callable=AsyncMock
            ) as mock_get_info,
        ):
            mock_manager.list_models.return_value = [sample_model_info]
            mock_get_info.return_value = {"loaded": True}

            result = await orchestration_service.load_model(
                model_name=sample_model_info["name"]
            )

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_load_model_not_found(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        異常系: 存在しないモデルの読み込み

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.list_models.return_value = []

            result = await orchestration_service.load_model(model_name="nonexistent")

            assert result["success"] is False


class TestGetCurrentModelInfo:
    """正常系: 現在のモデル情報取得のテスト"""

    @pytest.mark.asyncio
    async def test_get_current_model_info_success(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_metadata: Dict[str, Any],
    ):
        """
        正常系: 現在のモデル情報が正常に取得できる

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_metadata: サンプルメタデータ
        """
        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.os.path.exists"
            ) as mock_exists,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.os.path.getmtime"
            ) as mock_getmtime,
        ):
            mock_manager.get_latest_model.return_value = "/models/test.pkl"
            mock_exists.return_value = True
            mock_getmtime.return_value = 1640995200.0  # 2022-01-01 00:00:00
            mock_manager.load_model.return_value = {"metadata": sample_model_metadata}

            result = await orchestration_service.get_current_model_info()

            assert result["loaded"] is True
            assert result["is_trained"] is True

    @pytest.mark.asyncio
    async def test_get_current_model_info_no_model(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        エッジケース: モデルが存在しない場合

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.get_latest_model.return_value = None

            result = await orchestration_service.get_current_model_info()

            assert result["loaded"] is False


class TestErrorHandling:
    """異常系: エラーハンドリングのテスト"""

    @pytest.mark.asyncio
    async def test_update_ml_config_with_exception(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        異常系: 設定更新中に例外が発生した場合

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.ml_config_manager"
        ) as mock_config_manager:
            mock_config_manager.update_config.side_effect = Exception("Config error")

            result = await orchestration_service.update_ml_config(
                config_updates={"learning_rate": 0.05}
            )

            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_get_formatted_models_with_load_error(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_info: Dict[str, Any],
    ):
        """
        異常系: モデル読み込み中にエラーが発生した場合

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_info: サンプルモデル情報
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.list_models.return_value = [sample_model_info]
            mock_manager.load_model.side_effect = Exception("Load error")

            result = await orchestration_service.get_formatted_models()

            assert "models" in result
            assert len(result["models"]) == 1
            assert result["models"][0]["accuracy"] == 0.0


class TestEdgeCases:
    """境界値テスト"""

    @pytest.mark.asyncio
    async def test_get_feature_importance_with_top_n_zero(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        境界値: top_n=0の場合

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.os.path.exists"
            ) as mock_exists,
        ):
            mock_manager.get_latest_model.return_value = "/models/test.pkl"
            mock_exists.return_value = True
            mock_manager.load_model.return_value = {
                "metadata": {"feature_importance": {"feature1": 0.5}}
            }

            result = await orchestration_service.get_feature_importance(top_n=0)

            assert "feature_importance" in result
            assert len(result["feature_importance"]) == 0
