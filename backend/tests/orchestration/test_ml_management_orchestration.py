"""
ML管理オーケストレーションサービスのテストモジュール

MLManagementOrchestrationServiceの正常系、異常系、エッジケースをテストします。
"""

from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
from urllib.parse import quote

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
def sample_model_list() -> List[Dict[str, Any]]:
    """
    サンプルモデルリスト

    Returns:
        List[Dict[str, Any]]: モデル情報のリスト
    """
    return [
        {
            "name": "model_20240101_120000.pkl",
            "path": "/models/model_20240101_120000.pkl",
            "size_mb": 10.5,
            "modified_at": datetime(2024, 1, 1, 12, 0, 0),
            "directory": "/models",
        },
        {
            "name": "model_20240102_120000.pkl",
            "path": "/models/model_20240102_120000.pkl",
            "size_mb": 11.2,
            "modified_at": datetime(2024, 1, 2, 12, 0, 0),
            "directory": "/models",
        },
    ]


@pytest.fixture
def sample_model_metadata() -> Dict[str, Any]:
    """
    サンプルモデルメタデータ

    Returns:
        Dict[str, Any]: メタデータ辞書
    """
    return {
        "model_type": "LightGBM",
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "f1_score": 0.85,
        "auc_roc": 0.90,
        "feature_count": 50,
        "training_samples": 10000,
        "test_samples": 2500,
        "classification_report": {
            "0": {"precision": 0.80, "recall": 0.85, "f1-score": 0.82},
            "1": {"precision": 0.85, "recall": 0.90, "f1-score": 0.87},
            "macro avg": {"precision": 0.82, "recall": 0.88, "f1-score": 0.85},
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
        sample_model_list: List[Dict[str, Any]],
        sample_model_metadata: Dict[str, Any],
    ):
        """
        正常系: モデル一覧が正常に取得される

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_list: サンプルモデルリスト
            sample_model_metadata: サンプルメタデータ
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.list_models.return_value = sample_model_list
            mock_manager.load_model.return_value = {
                "metadata": sample_model_metadata
            }

            result = await orchestration_service.get_formatted_models()

            assert "models" in result
            assert len(result["models"]) == 2
            assert result["models"][0]["name"] == "model_20240101_120000.pkl"
            assert result["models"][0]["accuracy"] == 0.85

    @pytest.mark.asyncio
    async def test_get_formatted_models_empty(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        正常系: モデルが存在しない場合は空のリスト

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
    async def test_get_formatted_models_with_load_error(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_list: List[Dict[str, Any]],
    ):
        """
        正常系: モデル読み込みエラーでもデフォルト値を返す

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_list: サンプルモデルリスト
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.list_models.return_value = sample_model_list
            mock_manager.load_model.side_effect = Exception("Load error")

            result = await orchestration_service.get_formatted_models()

            assert "models" in result
            assert len(result["models"]) == 2
            # デフォルト値が設定される
            assert result["models"][0]["accuracy"] == 0.0
            assert result["models"][0]["model_type"] == "Unknown"


class TestDeleteModel:
    """正常系・異常系: モデル削除のテスト"""

    @pytest.mark.asyncio
    async def test_delete_model_success(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_list: List[Dict[str, Any]],
    ):
        """
        正常系: モデルが正常に削除される

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_list: サンプルモデルリスト
        """
        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch("os.path.exists", return_value=True),
            patch("os.remove") as mock_remove,
        ):
            mock_manager.list_models.return_value = sample_model_list

            result = await orchestration_service.delete_model(
                "model_20240101_120000.pkl"
            )

            assert result["success"] is True
            assert "削除" in result["message"]
            mock_remove.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_model_not_found(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_list: List[Dict[str, Any]],
    ):
        """
        異常系: 存在しないモデルで404エラー

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_list: サンプルモデルリスト
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.list_models.return_value = sample_model_list

            with pytest.raises(HTTPException) as exc_info:
                await orchestration_service.delete_model("nonexistent_model.pkl")

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_model_with_url_encoded_name(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_list: List[Dict[str, Any]],
    ):
        """
        正常系: URLエンコードされたモデル名でも削除できる

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_list: サンプルモデルリスト
        """
        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch("os.path.exists", return_value=True),
            patch("os.remove"),
        ):
            mock_manager.list_models.return_value = sample_model_list

            encoded_name = quote("model_20240101_120000.pkl")
            result = await orchestration_service.delete_model(encoded_name)

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_delete_model_file_not_exists(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_list: List[Dict[str, Any]],
    ):
        """
        異常系: モデルファイルが存在しない場合

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_list: サンプルモデルリスト
        """
        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch("os.path.exists", return_value=False),
        ):
            mock_manager.list_models.return_value = sample_model_list

            with pytest.raises(HTTPException) as exc_info:
                await orchestration_service.delete_model(
                    "model_20240101_120000.pkl"
                )

            assert exc_info.value.status_code == 404


class TestDeleteAllModels:
    """正常系: 全モデル削除のテスト"""

    @pytest.mark.asyncio
    async def test_delete_all_models_success(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_list: List[Dict[str, Any]],
    ):
        """
        正常系: 全モデルが正常に削除される

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_list: サンプルモデルリスト
        """
        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch("os.path.exists", return_value=True),
            patch("os.remove") as mock_remove,
        ):
            mock_manager.list_models.return_value = sample_model_list

            result = await orchestration_service.delete_all_models()

            assert result["success"] is True
            assert result["deleted_count"] == 2
            assert mock_remove.call_count == 2

    @pytest.mark.asyncio
    async def test_delete_all_models_empty(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        正常系: モデルが存在しない場合

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

    @pytest.mark.asyncio
    async def test_delete_all_models_with_failures(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_list: List[Dict[str, Any]],
    ):
        """
        正常系: 一部のモデル削除が失敗した場合

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_list: サンプルモデルリスト
        """
        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch("os.path.exists", return_value=True),
            patch("os.remove", side_effect=[None, Exception("Delete error")]),
        ):
            mock_manager.list_models.return_value = sample_model_list

            result = await orchestration_service.delete_all_models()

            assert result["success"] is True
            assert result["deleted_count"] == 1
            assert result["failed_count"] == 1
            assert len(result["failed_models"]) == 1


class TestGetMLStatus:
    """正常系: ML状態取得のテスト"""

    @pytest.mark.asyncio
    async def test_get_ml_status_with_model(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_metadata: Dict[str, Any],
    ):
        """
        正常系: モデルが存在する場合の状態取得

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_metadata: サンプルメタデータ
        """
        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch("os.path.exists", return_value=True),
            patch("os.path.getmtime", return_value=1704110400.0),
            patch("os.path.getsize", return_value=10485760),
        ):
            mock_manager.get_latest_model.return_value = "/models/latest.pkl"
            mock_manager.load_model.return_value = {
                "metadata": sample_model_metadata
            }

            result = await orchestration_service.get_ml_status()

            assert result["is_model_loaded"] is True
            assert result["is_trained"] is True
            assert result["model_type"] == "LightGBM"
            assert "model_info" in result
            assert "performance_metrics" in result

    @pytest.mark.asyncio
    async def test_get_ml_status_no_model(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        正常系: モデルが存在しない場合の状態取得

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.get_latest_model.return_value = None

            result = await orchestration_service.get_ml_status()

            assert result["is_model_loaded"] is False
            assert result["is_trained"] is False
            assert result["status"] == "no_model"
            assert "performance_metrics" in result

    @pytest.mark.asyncio
    async def test_get_ml_status_with_load_error(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        正常系: モデル読み込みエラーでもデフォルト状態を返す

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch("os.path.exists", return_value=True),
            patch("os.path.getmtime", return_value=1704110400.0),
            patch("os.path.getsize", return_value=10485760),
        ):
            mock_manager.get_latest_model.return_value = "/models/latest.pkl"
            mock_manager.load_model.side_effect = Exception("Load error")

            result = await orchestration_service.get_ml_status()

            assert result["is_model_loaded"] is False
            assert "model_info" in result
            assert result["model_info"]["accuracy"] == 0.0


class TestGetFeatureImportance:
    """正常系: 特徴量重要度取得のテスト"""

    @pytest.mark.asyncio
    async def test_get_feature_importance_success(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        正常系: 特徴量重要度が正常に取得される

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
            patch("os.path.exists", return_value=True),
        ):
            mock_manager.get_latest_model.return_value = "/models/latest.pkl"
            mock_manager.load_model.return_value = {
                "metadata": {"feature_importance": feature_importance}
            }

            result = await orchestration_service.get_feature_importance(top_n=10)

            assert "feature_importance" in result
            assert len(result["feature_importance"]) == 3

    @pytest.mark.asyncio
    async def test_get_feature_importance_top_n(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        正常系: トップN個の特徴量重要度を取得

        Args:
            orchestration_service: オーケストレーションサービス
        """
        feature_importance = {
            f"feature{i}": 1.0 / (i + 1) for i in range(20)
        }

        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch("os.path.exists", return_value=True),
        ):
            mock_manager.get_latest_model.return_value = "/models/latest.pkl"
            mock_manager.load_model.return_value = {
                "metadata": {"feature_importance": feature_importance}
            }

            result = await orchestration_service.get_feature_importance(top_n=5)

            assert "feature_importance" in result
            assert len(result["feature_importance"]) == 5

    @pytest.mark.asyncio
    async def test_get_feature_importance_no_model(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        正常系: モデルが存在しない場合は空のリスト

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.get_latest_model.return_value = None

            result = await orchestration_service.get_feature_importance()

            assert "feature_importance" in result
            assert result["feature_importance"] == []


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
            result = await orchestration_service.cleanup_old_models()

            assert "message" in result
            assert "削除" in result["message"]
            mock_manager.cleanup_expired_models.assert_called_once()


class TestLoadModel:
    """正常系・異常系: モデル読み込みのテスト"""

    @pytest.mark.asyncio
    async def test_load_model_success(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_list: List[Dict[str, Any]],
    ):
        """
        正常系: モデルが正常に読み込まれる

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_list: サンプルモデルリスト
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.list_models.return_value = sample_model_list

            result = await orchestration_service.load_model(
                "model_20240101_120000.pkl"
            )

            assert result["success"] is True
            assert "読み込み" in result["message"]

    @pytest.mark.asyncio
    async def test_load_model_not_found(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_list: List[Dict[str, Any]],
    ):
        """
        異常系: 存在しないモデルの読み込み

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_list: サンプルモデルリスト
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.list_models.return_value = sample_model_list

            result = await orchestration_service.load_model(
                "nonexistent_model.pkl"
            )

            assert result["success"] is False
            assert "見つかりません" in result["error"]


class TestGetCurrentModelInfo:
    """正常系: 現在のモデル情報取得のテスト"""

    @pytest.mark.asyncio
    async def test_get_current_model_info_success(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_metadata: Dict[str, Any],
    ):
        """
        正常系: 現在のモデル情報が正常に取得される

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_metadata: サンプルメタデータ
        """
        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch("os.path.exists", return_value=True),
            patch("os.path.getmtime", return_value=1704110400.0),
        ):
            mock_manager.get_latest_model.return_value = "/models/latest.pkl"
            mock_manager.load_model.return_value = {
                "metadata": sample_model_metadata
            }

            result = await orchestration_service.get_current_model_info()

            assert result["loaded"] is True
            assert result["is_trained"] is True
            assert result["model_type"] == "LightGBM"

    @pytest.mark.asyncio
    async def test_get_current_model_info_no_model(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        正常系: モデルが存在しない場合

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.get_latest_model.return_value = None

            result = await orchestration_service.get_current_model_info()

            assert result["loaded"] is False
            assert "見つかりません" in result["message"]


class TestMLConfigManagement:
    """正常系: ML設定管理のテスト"""

    def test_get_ml_config_dict(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        正常系: ML設定が辞書形式で取得される

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.ml_config_manager"
        ) as mock_manager:
            mock_manager.get_config_dict.return_value = {"key": "value"}

            result = orchestration_service.get_ml_config_dict()

            assert result == {"key": "value"}
            mock_manager.get_config_dict.assert_called_once()

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
        ) as mock_manager:
            mock_manager.update_config.return_value = True
            mock_manager.get_config_dict.return_value = {"updated": "config"}

            result = await orchestration_service.update_ml_config(
                {"key": "new_value"}
            )

            assert result["success"] is True
            assert "更新されました" in result["message"]

    @pytest.mark.asyncio
    async def test_update_ml_config_failure(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        異常系: ML設定の更新が失敗

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.ml_config_manager"
        ) as mock_manager:
            mock_manager.update_config.return_value = False

            result = await orchestration_service.update_ml_config(
                {"key": "new_value"}
            )

            assert result["success"] is False
            assert "失敗" in result["message"]

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
        ) as mock_manager:
            mock_manager.reset_config.return_value = True
            mock_manager.get_config_dict.return_value = {"default": "config"}

            result = await orchestration_service.reset_ml_config()

            assert result["success"] is True
            assert "リセット" in result["message"]

    @pytest.mark.asyncio
    async def test_reset_ml_config_failure(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        異常系: ML設定のリセットが失敗

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.ml_config_manager"
        ) as mock_manager:
            mock_manager.reset_config.return_value = False

            result = await orchestration_service.reset_ml_config()

            assert result["success"] is False
            assert "失敗" in result["message"]


class TestIsActiveModel:
    """正常系: アクティブモデル判定のテスト"""

    def test_is_active_model_true(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_list: List[Dict[str, Any]],
    ):
        """
        正常系: 最新モデルがアクティブと判定される

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_list: サンプルモデルリスト
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.get_latest_model.return_value = sample_model_list[0][
                "path"
            ]

            result = orchestration_service._is_active_model(sample_model_list[0])

            assert result is True

    def test_is_active_model_false(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_list: List[Dict[str, Any]],
    ):
        """
        正常系: 古いモデルが非アクティブと判定される

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_list: サンプルモデルリスト
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.get_latest_model.return_value = sample_model_list[0][
                "path"
            ]

            result = orchestration_service._is_active_model(sample_model_list[1])

            assert result is False

    def test_is_active_model_single_model(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_list: List[Dict[str, Any]],
    ):
        """
        正常系: 単一モデルの場合はアクティブと判定される

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_list: サンプルモデルリスト
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.get_latest_model.return_value = None
            mock_manager.list_models.return_value = [sample_model_list[0]]

            result = orchestration_service._is_active_model(sample_model_list[0])

            assert result is True


class TestEdgeCases:
    """境界値テスト"""

    @pytest.mark.asyncio
    async def test_get_feature_importance_with_zero_top_n(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        境界値: top_n=0の場合

        Args:
            orchestration_service: オーケストレーションサービス
        """
        feature_importance = {"feature1": 0.5, "feature2": 0.3}

        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch("os.path.exists", return_value=True),
        ):
            mock_manager.get_latest_model.return_value = "/models/latest.pkl"
            mock_manager.load_model.return_value = {
                "metadata": {"feature_importance": feature_importance}
            }

            result = await orchestration_service.get_feature_importance(top_n=0)

            assert "feature_importance" in result
            assert len(result["feature_importance"]) == 0

    @pytest.mark.asyncio
    async def test_delete_model_with_special_characters(
        self,
        orchestration_service: MLManagementOrchestrationService,
    ):
        """
        境界値: 特殊文字を含むモデル名

        Args:
            orchestration_service: オーケストレーションサービス
        """
        special_model = {
            "name": "model-test_2024@01#01.pkl",
            "path": "/models/model-test_2024@01#01.pkl",
            "size_mb": 10.0,
            "modified_at": datetime(2024, 1, 1),
            "directory": "/models",
        }

        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch("os.path.exists", return_value=True),
            patch("os.remove"),
        ):
            mock_manager.list_models.return_value = [special_model]

            result = await orchestration_service.delete_model(
                "model-test_2024@01#01.pkl"
            )

            assert result["success"] is True