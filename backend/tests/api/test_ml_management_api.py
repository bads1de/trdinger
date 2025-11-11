"""
ML管理APIのテストモジュール

ML管理APIエンドポイントの正常系、異常系、エッジケースをテストします。
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.dependencies import get_ml_management_orchestration_service, get_db


@pytest.fixture
def test_client() -> TestClient:
    """
    FastAPIテストクライアントのフィクスチャ

    Returns:
        TestClient: テスト用のFastAPIクライアント
    """
    return TestClient(app)

@pytest.fixture
def mock_ml_management_orchestration_service() -> AsyncMock:
    """
    オーケストレーションサービスのモック

    Returns:
        AsyncMock: モックされたオーケストレーションサービス
    """
    return AsyncMock()


@pytest.fixture
def mock_db_session() -> Mock:
    """
    データベースセッションのモック

    Returns:
        Mock: モックされたデータベースセッション
    """
    return Mock()


@pytest.fixture
def mock_ml_management_orchestration_service() -> Mock:
    """
    MLManagementOrchestrationServiceのモック

    Returns:
        Mock: モックされたML管理オーケストレーションサービス
    """
    mock_service = Mock()
    mock_service.get_formatted_models = AsyncMock()
    mock_service.delete_model = AsyncMock()
    mock_service.delete_all_models = AsyncMock()
    mock_service.get_ml_status = AsyncMock()
    mock_service.get_feature_importance = AsyncMock()
    mock_service.load_model = AsyncMock()
    mock_service.get_current_model_info = AsyncMock()
    mock_service.get_ml_config_dict = Mock()
    mock_service.update_ml_config = AsyncMock()
    mock_service.reset_ml_config = AsyncMock()
    mock_service.cleanup_old_models = AsyncMock()
    return mock_service


@pytest.fixture(autouse=True)
def override_dependencies(mock_db_session, mock_ml_management_orchestration_service):
    """
    FastAPIの依存性注入をオーバーライド
    
    Args:
        mock_db_session: モックDBセッション
        mock_ml_management_orchestration_service: モックサービス
    """
    app.dependency_overrides[get_db] = lambda: mock_db_session
    app.dependency_overrides[get_ml_management_orchestration_service] = lambda: mock_ml_management_orchestration_service
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def sample_model_info() -> Dict[str, Any]:
    """
    サンプルモデル情報

    Returns:
        Dict[str, Any]: モデル情報のサンプル
    """
    return {
        "model_id": "model_12345",
        "model_name": "ensemble_model_2024_01_01",
        "model_type": "ensemble",
        "created_at": "2024-01-01T00:00:00",
        "metrics": {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.87,
            "f1_score": 0.85,
        },
    }


@pytest.fixture
def sample_models_list(sample_model_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    サンプルモデルリスト

    Args:
        sample_model_info: 単一のモデル情報

    Returns:
        List[Dict[str, Any]]: モデル情報のリスト
    """
    return [
        sample_model_info,
        {**sample_model_info, "model_id": "model_12346", "model_type": "lightgbm"},
        {**sample_model_info, "model_id": "model_12347", "model_type": "xgboost"},
    ]


class TestGetModels:
    """モデル一覧取得のテストクラス"""
    def test_get_models_success(
        self,
        test_client: TestClient,
        mock_ml_management_orchestration_service: AsyncMock,
        sample_models_list: List[Dict[str, Any]],
    ) -> None:
        """
        正常系: モデル一覧が正常に取得できる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_ml_management_orchestration_service: オーケストレーションサービスモック
            sample_models_list: サンプルモデルリスト
        """
        # モックの設定
        mock_ml_management_orchestration_service.get_formatted_models.return_value = {
            "success": True,
            "data": {"models": sample_models_list, "count": len(sample_models_list)},
        }

        # APIリクエスト
        response = test_client.get("/api/ml/models")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["models"]) == 3
    def test_get_models_empty(
        self,
        test_client: TestClient,
        mock_ml_management_orchestration_service: AsyncMock,
    ) -> None:
        """
        エッジケース: モデルが空の場合

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_ml_management_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_ml_management_orchestration_service.get_formatted_models.return_value = {
            "success": True,
            "data": {"models": [], "count": 0},
        }

        # APIリクエスト
        response = test_client.get("/api/ml/models")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["models"]) == 0


class TestDeleteModels:
    """モデル削除のテストクラス"""
    def test_delete_model_success(
        self,
        test_client: TestClient,
        mock_ml_management_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: モデルが正常に削除できる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_ml_management_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_ml_management_orchestration_service.delete_model.return_value = {
            "success": True,
            "message": "モデルを削除しました",
        }

        # APIリクエスト
        response = test_client.delete("/api/ml/models/model_12345")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "削除" in data["message"]
    def test_delete_model_not_found(
        self,
        test_client: TestClient,
        mock_ml_management_orchestration_service: AsyncMock,
    ) -> None:
        """
        異常系: 存在しないモデルの削除

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_ml_management_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_ml_management_orchestration_service.delete_model.return_value = {
            "success": False,
            "message": "モデルが見つかりません",
            "status_code": 404,
        }

        # APIリクエスト
        response = test_client.delete("/api/ml/models/nonexistent_model")

        # アサーション
        assert response.status_code in [200, 404]
    def test_delete_all_models_success(
        self,
        test_client: TestClient,
        mock_ml_management_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: すべてのモデルが正常に削除できる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_ml_management_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_ml_management_orchestration_service.delete_all_models.return_value = {
            "success": True,
            "message": "すべてのモデルを削除しました",
            "data": {"deleted_count": 5},
        }

        # APIリクエスト
        response = test_client.delete("/api/ml/models/all")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["deleted_count"] == 5


class TestModelStatus:
    """モデル状態取得のテストクラス"""
    def test_get_ml_status_success(
        self,
        test_client: TestClient,
        mock_ml_management_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: ML状態が正常に取得できる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_ml_management_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_ml_management_orchestration_service.get_ml_status.return_value = {
            "success": True,
            "data": {
                "model_count": 5,
                "current_model": "model_12345",
                "last_training": "2024-01-01T00:00:00",
            },
        }

        # APIリクエスト
        response = test_client.get("/api/ml/status")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["model_count"] == 5


class TestFeatureImportance:
    """特徴量重要度取得のテストクラス"""
    def test_get_feature_importance_success(
        self,
        test_client: TestClient,
        mock_ml_management_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: 特徴量重要度が正常に取得できる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_ml_management_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_ml_management_orchestration_service.get_feature_importance.return_value = {
            "success": True,
            "data": {
                "features": [
                    {"name": "rsi", "importance": 0.25},
                    {"name": "macd", "importance": 0.20},
                ]
            },
        }

        # APIリクエスト
        response = test_client.get("/api/ml/feature-importance", params={"top_n": 10})

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["features"]) == 2
    def test_get_feature_importance_no_model(
        self,
        test_client: TestClient,
        mock_ml_management_orchestration_service: AsyncMock,
    ) -> None:
        """
        エッジケース: モデルが存在しない場合

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_ml_management_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_ml_management_orchestration_service.get_feature_importance.return_value = {
            "success": False,
            "message": "モデルが見つかりません",
        }

        # APIリクエスト
        response = test_client.get("/api/ml/feature-importance")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


class TestLoadModel:
    """モデル読み込みのテストクラス"""
    def test_load_model_success(
        self,
        test_client: TestClient,
        mock_ml_management_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: モデルが正常に読み込まれる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_ml_management_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_ml_management_orchestration_service.load_model.return_value = {
            "success": True,
            "message": "モデルを読み込みました",
        }

        # APIリクエスト
        response = test_client.post("/api/ml/models/model_12345/load")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    def test_get_current_model_success(
        self,
        test_client: TestClient,
        mock_ml_management_orchestration_service: AsyncMock,
        sample_model_info: Dict[str, Any],
    ) -> None:
        """
        正常系: 現在のモデル情報が取得できる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_ml_management_orchestration_service: オーケストレーションサービスモック
            sample_model_info: サンプルモデル情報
        """
        # モックの設定
        mock_ml_management_orchestration_service.get_current_model_info.return_value = {
            "success": True,
            "data": sample_model_info,
        }

        # APIリクエスト
        response = test_client.get("/api/ml/models/current")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["model_id"] == "model_12345"


class TestMLConfig:
    """ML設定のテストクラス"""
    def test_get_ml_config_success(
        self,
        test_client: TestClient,
        mock_ml_management_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: ML設定が正常に取得できる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_ml_management_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_ml_management_orchestration_service.get_ml_config_dict.return_value = {
            "success": True,
            "data": {
                "model_type": "ensemble",
                "learning_rate": 0.1,
                "max_depth": 10,
            },
        }

        # APIリクエスト
        response = test_client.get("/api/ml/config")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
    def test_update_ml_config_success(
        self,
        test_client: TestClient,
        mock_ml_management_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: ML設定が正常に更新できる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_ml_management_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_ml_management_orchestration_service.update_ml_config.return_value = {
            "success": True,
            "message": "設定を更新しました",
            "updated_config": {"learning_rate": 0.05},
        }

        # APIリクエスト
        config_data = {"learning_rate": 0.05}
        response = test_client.put("/api/ml/config", json=config_data)

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    def test_reset_ml_config_success(
        self,
        test_client: TestClient,
        mock_ml_management_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: ML設定が正常にリセットできる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_ml_management_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_ml_management_orchestration_service.reset_ml_config.return_value = {
            "success": True,
            "message": "設定をリセットしました",
            "config": {"learning_rate": 0.1},
        }

        # APIリクエスト
        response = test_client.post("/api/ml/config/reset")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestCleanupModels:
    """モデルクリーンアップのテストクラス"""
    def test_cleanup_old_models_success(
        self,
        test_client: TestClient,
        mock_ml_management_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: 古いモデルが正常にクリーンアップできる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_ml_management_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_ml_management_orchestration_service.cleanup_old_models.return_value = {
            "success": True,
            "message": "古いモデルをクリーンアップしました",
            "data": {"deleted_count": 3},
        }

        # APIリクエスト
        response = test_client.post("/api/ml/models/cleanup")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["deleted_count"] == 3


class TestErrorHandling:
    """エラーハンドリングのテストクラス"""
    def test_service_error_handling(
        self,
        test_client: TestClient,
        mock_ml_management_orchestration_service: AsyncMock,
    ) -> None:
        """
        異常系: サービス層でエラーが発生した場合

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_ml_management_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_ml_management_orchestration_service.get_formatted_models.side_effect = (
            Exception("Service error")
        )

        # APIリクエスト
        response = test_client.get("/api/ml/models")

        # アサーション（ErrorHandlerによって処理される）
        assert response.status_code in [200, 500]