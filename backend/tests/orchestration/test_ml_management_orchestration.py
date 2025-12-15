"""
ML管理オーケストレーションサービスのテストモジュール

MLManagementOrchestrationServiceの正常系、異常系、エッジケースをテストします。
"""

from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import patch
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
        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.load_model_metadata_safely"
            ) as mock_load_metadata,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.ml_training_service"
            ) as mock_training_service,
        ):
            mock_manager.list_models.return_value = sample_model_list
            mock_load_metadata.return_value = {"metadata": sample_model_metadata}
            mock_manager.extract_model_performance_metrics.return_value = {
                "accuracy": 0.85
            }
            # アクティブモデルの設定
            mock_training_service.get_current_model_path.return_value = (
                "/models/model_20240101_120000.pkl"
            )

            result = await orchestration_service.get_formatted_models()

            assert "models" in result
            assert len(result["models"]) == 2
            assert result["models"][0]["name"] == "model_20240101_120000.pkl"
            assert result["models"][0]["accuracy"] == 0.85
            # アクティブモデルの検証
            assert result["models"][0]["is_active"] is True
            assert result["models"][1]["is_active"] is False

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
        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.load_model_metadata_safely"
            ) as mock_load_metadata,
        ):
            mock_manager.list_models.return_value = sample_model_list
            mock_load_metadata.side_effect = Exception("Load error")

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
                await orchestration_service.delete_model("model_20240101_120000.pkl")

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_model_race_condition(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_list: List[Dict[str, Any]],
    ):
        """
        異常系: ファイル削除時にFileNotFoundErrorが発生した場合

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_list: サンプルモデルリスト
        """
        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch("os.remove", side_effect=FileNotFoundError("File not found")),
        ):
            mock_manager.list_models.return_value = sample_model_list

            # FileNotFoundErrorがHTTPException 404として処理されることを確認
            with pytest.raises(HTTPException) as exc_info:
                await orchestration_service.delete_model("model_20240101_120000.pkl")

            assert exc_info.value.status_code == 404
            assert "存在しません" in exc_info.value.detail


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

    @pytest.mark.asyncio
    async def test_delete_all_models_handles_missing_files_gracefully(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        異常系: 削除処理中にファイルが消えてもエラーにならない（レースコンディション対策）
        """
        from datetime import datetime

        sample_models = [
            {
                "name": "model1.pkl",
                "path": "/models/model1.pkl",
                "size_mb": 10.0,
                "modified_at": datetime(2024, 1, 1),
                "directory": "/models",
            },
            {
                "name": "model2.pkl",
                "path": "/models/model2.pkl",
                "size_mb": 10.0,
                "modified_at": datetime(2024, 1, 1),
                "directory": "/models",
            },
        ]

        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch("os.remove", side_effect=[None, FileNotFoundError("File deleted")]),
        ):
            mock_manager.list_models.return_value = sample_models

            result = await orchestration_service.delete_all_models()

            # 1つ成功、1つ失敗
            assert result["success"] is True
            assert result["deleted_count"] == 1
            assert result["failed_count"] == 1
            assert "model2.pkl" in result["failed_models"]


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
        from datetime import datetime

        model_info_data = {
            "path": "/models/latest.pkl",
            "metadata": sample_model_metadata,
            "metrics": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85,
            },
            "file_info": {
                "size_mb": 10.0,
                "modified_at": datetime(2024, 1, 1, 12, 0, 0),
            },
        }

        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.get_latest_model_with_info"
        ) as mock_get_model:
            mock_get_model.return_value = model_info_data

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
            "app.services.ml.orchestration.ml_management_orchestration_service.get_latest_model_with_info"
        ) as mock_get_model:
            mock_get_model.return_value = None

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

    @pytest.mark.asyncio
    async def test_get_ml_status_with_missing_metrics_key(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_metadata: Dict[str, Any],
    ):
        """
        異常系: model_info_dataにmetricsキーがない場合でもエラーにならない

        Args:
            orchestration_service: オーケストレーションサービス
            sample_model_metadata: サンプルメタデータ
        """
        from datetime import datetime

        # metricsキーを含まないmodel_info_data
        model_info_data_without_metrics = {
            "path": "/models/latest.pkl",
            "metadata": sample_model_metadata,
            # "metrics": {}  <- 意図的にmetricsキーを省略
            "file_info": {
                "size_mb": 10.0,
                "modified_at": datetime(2024, 1, 1, 12, 0, 0),
            },
        }

        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.get_latest_model_with_info"
        ) as mock_get_model:
            mock_get_model.return_value = model_info_data_without_metrics

            # KeyErrorが発生しないことを確認
            result = await orchestration_service.get_ml_status()

            assert result["is_model_loaded"] is True
            assert "performance_metrics" in result
            # デフォルトメトリクスが使用される
            assert "accuracy" in result["performance_metrics"]

    @pytest.mark.asyncio
    async def test_get_ml_status_metrics_consistency(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        正常系: メトリクスが二重取得されず、一貫性があることを確認
        """
        from datetime import datetime

        sample_metadata = {
            "model_type": "LightGBM",
            "feature_count": 50,
            "training_samples": 10000,
        }

        model_info_data = {
            "path": "/models/latest.pkl",
            "metadata": sample_metadata,
            "metrics": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85,
            },
            "file_info": {
                "size_mb": 10.0,
                "modified_at": datetime(2024, 1, 1, 12, 0, 0),
            },
        }

        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.get_latest_model_with_info"
        ) as mock_get_model:
            mock_get_model.return_value = model_info_data

            result = await orchestration_service.get_ml_status()

            # performance_metricsとmodel_info内のメトリクスが一致することを確認
            assert result["performance_metrics"]["accuracy"] == 0.85
            assert result["model_info"]["accuracy"] == 0.85


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
        from datetime import datetime

        feature_importance = {
            "feature1": 0.5,
            "feature2": 0.3,
            "feature3": 0.2,
        }

        model_info_data = {
            "path": "/models/latest.pkl",
            "metadata": {"feature_importance": feature_importance},
            "metrics": {},
            "file_info": {"size_mb": 10.0, "modified_at": datetime(2024, 1, 1)},
        }

        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.get_latest_model_with_info"
        ) as mock_get_model:
            mock_get_model.return_value = model_info_data

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
        from datetime import datetime

        feature_importance = {f"feature{i}": 1.0 / (i + 1) for i in range(20)}

        model_info_data = {
            "path": "/models/latest.pkl",
            "metadata": {"feature_importance": feature_importance},
            "metrics": {},
            "file_info": {"size_mb": 10.0, "modified_at": datetime(2024, 1, 1)},
        }

        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.get_latest_model_with_info"
        ) as mock_get_model:
            mock_get_model.return_value = model_info_data

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
            "app.services.ml.orchestration.ml_management_orchestration_service.get_latest_model_with_info"
        ) as mock_get_model:
            mock_get_model.return_value = None

            result = await orchestration_service.get_feature_importance()

            assert "feature_importance" in result
            assert result["feature_importance"] == {}

    @pytest.mark.asyncio
    async def test_feature_importance_returns_dict_when_no_importance_data(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        正常系: feature_importanceがメタデータに存在しない場合も辞書を返す
        """
        from datetime import datetime

        model_info_data = {
            "path": "/models/latest.pkl",
            "metadata": {},  # feature_importanceなし
            "metrics": {},
            "file_info": {"size_mb": 10.0, "modified_at": datetime(2024, 1, 1)},
        }

        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.get_latest_model_with_info"
        ) as mock_get_model:
            mock_get_model.return_value = model_info_data

            result = await orchestration_service.get_feature_importance()

            assert isinstance(result["feature_importance"], dict)
            assert result["feature_importance"] == {}

    @pytest.mark.asyncio
    async def test_feature_importance_returns_dict_when_load_error(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        異常系: 読み込みエラー時も辞書を返す
        """
        from datetime import datetime

        model_info_data = {
            "path": "/models/latest.pkl",
            "metadata": {"feature_importance": "invalid"},  # 不正なデータ
            "metrics": {},
            "file_info": {"size_mb": 10.0, "modified_at": datetime(2024, 1, 1)},
        }

        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.get_latest_model_with_info"
        ) as mock_get_model:
            mock_get_model.return_value = model_info_data

            result = await orchestration_service.get_feature_importance()

            assert isinstance(result["feature_importance"], dict)
            assert result["feature_importance"] == {}


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
        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.ml_training_service"
            ) as mock_training_service,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.MLManagementOrchestrationService.get_current_model_info"
            ) as mock_get_info,
        ):
            mock_manager.list_models.return_value = sample_model_list
            mock_training_service.load_model.return_value = True
            mock_get_info.return_value = {"loaded": True}

            result = await orchestration_service.load_model("model_20240101_120000.pkl")

            assert result["success"] is True
            assert "読み込み" in result["message"]

    @pytest.mark.asyncio
    async def test_load_model_calls_training_service(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_list: List[Dict[str, Any]],
    ):
        """
        正常系: モデル読み込み時にMLTrainingService.load_modelが呼び出される
        """
        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.ml_training_service"
            ) as mock_training_service,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.MLManagementOrchestrationService.get_current_model_info"
            ) as mock_get_info,
        ):
            # Setup mocks
            mock_manager.list_models.return_value = sample_model_list
            mock_training_service.load_model.return_value = True
            mock_get_info.return_value = {"loaded": True}

            # Execute
            result = await orchestration_service.load_model("model_20240101_120000.pkl")

            # Verify
            assert result["success"] is True
            mock_training_service.load_model.assert_called_once_with(
                "/models/model_20240101_120000.pkl"
            )

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

            result = await orchestration_service.load_model("nonexistent_model.pkl")

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
        """
        from datetime import datetime

        current_metadata = {
            **sample_model_metadata,
            "metrics": {"accuracy": 0.85},
        }

        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.ml_training_service"
            ) as mock_training_service,
            patch("os.path.exists", return_value=True),
            patch("os.stat") as mock_stat,
        ):

            mock_training_service.get_current_model_path.return_value = (
                "/models/latest.pkl"
            )
            mock_training_service.get_current_model_info.return_value = current_metadata

            mock_stat.return_value.st_mtime = datetime(2024, 1, 1, 12, 0, 0).timestamp()

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
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.ml_training_service"
        ) as mock_training_service:
            mock_training_service.get_current_model_path.return_value = None
            mock_training_service.get_current_model_info.return_value = None

            result = await orchestration_service.get_current_model_info()

            assert result["loaded"] is False
            assert "ロードされていません" in result["message"]

    @pytest.mark.asyncio
    async def test_get_current_model_info_with_classification_report_only(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        正常系: classification_reportのみ存在する場合メトリクスを抽出する

        Args:
            orchestration_service: オーケストレーションサービス
        """
        from datetime import datetime

        # metricsキーなし、classification_reportのみ
        current_metadata = {
            "model_type": "LightGBM",
            "feature_count": 50,
            "training_samples": 10000,
            "classification_report": {
                "0": {"precision": 0.80, "recall": 0.85, "f1-score": 0.82},
                "1": {"precision": 0.85, "recall": 0.90, "f1-score": 0.87},
                "accuracy": 0.88,
                "macro avg": {"precision": 0.82, "recall": 0.88, "f1-score": 0.85},
            },
        }

        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.ml_training_service"
            ) as mock_training_service,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch("os.path.exists", return_value=True),
            patch("os.stat") as mock_stat,
        ):
            mock_training_service.get_current_model_path.return_value = (
                "/models/latest.pkl"
            )
            mock_training_service.get_current_model_info.return_value = current_metadata

            # extract_model_performance_metricsがclassification_reportから抽出
            mock_manager.extract_model_performance_metrics.return_value = {
                "accuracy": 0.88,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85,
            }

            mock_stat.return_value.st_mtime = datetime(2024, 1, 1, 12, 0, 0).timestamp()

            result = await orchestration_service.get_current_model_info()

            assert result["loaded"] is True
            assert result["accuracy"] == 0.88
            assert result["precision"] == 0.82
            # extract_model_performance_metricsが呼び出されたことを確認
            mock_manager.extract_model_performance_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_current_model_info_with_extract_error(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        異常系: extract_model_performance_metricsが例外を投げてもデフォルト値を使用
        """
        from datetime import datetime

        current_metadata = {
            "model_type": "LightGBM",
            "feature_count": 50,
            "training_samples": 10000,
        }

        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.ml_training_service"
            ) as mock_training_service,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch("os.path.exists", return_value=True),
            patch("os.stat") as mock_stat,
        ):
            mock_training_service.get_current_model_path.return_value = (
                "/models/latest.pkl"
            )
            mock_training_service.get_current_model_info.return_value = current_metadata

            # extract_model_performance_metricsが例外を投げる
            mock_manager.extract_model_performance_metrics.side_effect = Exception(
                "Extraction error"
            )

            mock_stat.return_value.st_mtime = datetime(2024, 1, 1, 12, 0, 0).timestamp()

            result = await orchestration_service.get_current_model_info()

            # エラーが発生してもデフォルト値を返すべき
            assert result["loaded"] is True
            assert result["accuracy"] == 0.0  # デフォルト値


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

            result = await orchestration_service.update_ml_config({"key": "new_value"})

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

            result = await orchestration_service.update_ml_config({"key": "new_value"})

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
        正常系: ロードされているモデルとパスが一致する場合
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.ml_training_service"
        ) as mock_training_service:
            mock_training_service.get_current_model_path.return_value = (
                sample_model_list[0]["path"]
            )

            result = orchestration_service._is_active_model(sample_model_list[0])

            assert result is True

    def test_is_active_model_false(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_list: List[Dict[str, Any]],
    ):
        """
        正常系: ロードされているモデルとパスが一致しない場合
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.ml_training_service"
        ) as mock_training_service:
            mock_training_service.get_current_model_path.return_value = (
                "/models/other_model.pkl"
            )

            result = orchestration_service._is_active_model(sample_model_list[0])

            assert result is False

    def test_is_active_model_none(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_list: List[Dict[str, Any]],
    ):
        """
        正常系: モデルがロードされていない場合
        """
        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.ml_training_service"
        ) as mock_training_service:
            mock_training_service.get_current_model_path.return_value = None

            result = orchestration_service._is_active_model(sample_model_list[0])

            assert result is False

    def test_is_active_model_handles_exceptions(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        異常系: 例外発生時はFalseを返す
        """
        from datetime import datetime

        sample_model = {
            "name": "test.pkl",
            "path": "/models/test.pkl",
            "size_mb": 10.0,
            "modified_at": datetime(2024, 1, 1),
            "directory": "/models",
        }

        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.ml_training_service"
        ) as mock_service:
            # AttributeError
            mock_service.get_current_model_path.side_effect = AttributeError(
                "Method not found"
            )
            assert orchestration_service._is_active_model(sample_model) is False

            # TypeError
            mock_service.get_current_model_path.side_effect = TypeError("Type mismatch")
            assert orchestration_service._is_active_model(sample_model) is False


class TestEdgeCases:
    """エッジケースのテスト"""

    @pytest.mark.asyncio
    async def test_get_feature_importance_with_zero_top_n(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        エッジケース: top_n=0の場合
        """
        from datetime import datetime

        feature_importance = {"feature1": 0.5, "feature2": 0.3}

        model_info_data = {
            "path": "/models/latest.pkl",
            "metadata": {"feature_importance": feature_importance},
            "metrics": {},
            "file_info": {"size_mb": 10.0, "modified_at": datetime(2024, 1, 1)},
        }

        with patch(
            "app.services.ml.orchestration.ml_management_orchestration_service.get_latest_model_with_info"
        ) as mock_get_model:
            mock_get_model.return_value = model_info_data

            result = await orchestration_service.get_feature_importance(top_n=0)

            assert "feature_importance" in result
            assert result["feature_importance"] == {}

    @pytest.mark.asyncio
    async def test_delete_model_with_special_characters(
        self,
        orchestration_service: MLManagementOrchestrationService,
        sample_model_list: List[Dict[str, Any]],
    ):
        """
        エッジケース: 特殊文字を含むモデル名の削除
        """
        from datetime import datetime

        # 特殊文字を含むモデルをリストに追加
        special_model = {
            "name": "model name with spaces.pkl",
            "path": "/models/model name with spaces.pkl",
            "size_mb": 10.0,
            "modified_at": datetime(2024, 1, 1),
            "directory": "/models",
        }
        models_list = sample_model_list + [special_model]

        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch("os.path.exists", return_value=True),
            patch("os.remove"),
        ):
            mock_manager.list_models.return_value = models_list

            # スペースや特殊文字を含む名前
            special_name = quote("model name with spaces.pkl")
            result = await orchestration_service.delete_model(special_name)

            assert result["success"] is True




