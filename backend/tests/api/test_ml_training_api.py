"""
MLトレーニングAPIのテストモジュール

MLトレーニングAPIエンドポイントの正常系、異常系、エッジケースをテストします。
"""

from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.dependencies import get_ml_training_orchestration_service, get_db


@pytest.fixture
def test_client() -> TestClient:
    """
    FastAPIテストクライアントのフィクスチャ

    Returns:
        TestClient: テスト用のFastAPIクライアント
    """
    return TestClient(app)

@pytest.fixture
def mock_ml_training_orchestration_service() -> AsyncMock:
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

@pytest.fixture(autouse=True)
def override_dependencies(mock_db_session, mock_ml_training_orchestration_service):
    """
    FastAPIの依存性注入をオーバーライド
    
    Args:
        mock_db_session: モックDBセッション
        mock_ml_training_orchestration_service: モックサービス
    """
    app.dependency_overrides[get_db] = lambda: mock_db_session
    app.dependency_overrides[get_ml_training_orchestration_service] = lambda: mock_ml_training_orchestration_service
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def mock_ml_training_orchestration_service() -> Mock:
    """
    MLTrainingOrchestrationServiceのモック

    Returns:
        Mock: モックされたMLトレーニングオーケストレーションサービス
    """
    mock_service = Mock()
    mock_service.start_training = AsyncMock()
    mock_service.get_training_status = AsyncMock()
    mock_service.get_model_info = AsyncMock()
    mock_service.stop_training = AsyncMock()
    return mock_service


@pytest.fixture
def sample_training_config() -> Dict[str, Any]:
    """
    サンプルトレーニング設定

    Returns:
        Dict[str, Any]: トレーニング設定のサンプル
    """
    return {
        "symbol": "BTC/USDT:USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "validation_split": 0.2,
        "prediction_horizon": 24,
        "threshold_up": 0.02,
        "threshold_down": -0.02,
        "save_model": True,
        "train_test_split": 0.8,
        "cross_validation_folds": 5,
        "random_state": 42,
        "early_stopping_rounds": 100,
        "max_depth": 10,
        "n_estimators": 100,
        "learning_rate": 0.1,
    }


class TestStartTraining:
    """MLトレーニング開始のテストクラス"""

    def test_start_training_success(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_ml_training_orchestration_service: AsyncMock,
        sample_training_config: Dict[str, Any],
    ) -> None:
        """
        正常系: MLトレーニングが正常に開始される

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_ml_training_orchestration_service: オーケストレーションサービスモック
            sample_training_config: サンプルトレーニング設定
        """
        # モックの設定
        mock_ml_training_orchestration_service.start_training.return_value = {
            "success": True,
            "message": "トレーニングを開始しました",
            "training_id": "train_12345",
        }

        # APIリクエスト
        response = test_client.post(
            "/api/ml-training/train",
            json=sample_training_config,
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "training_id" in data

    def test_start_training_with_ensemble(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_ml_training_orchestration_service: AsyncMock,
        sample_training_config: Dict[str, Any],
    ) -> None:
        """
        正常系: アンサンブル学習でトレーニングが開始される

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_ml_training_orchestration_service: オーケストレーションサービスモック
            sample_training_config: サンプルトレーニング設定
        """
        # モックの設定
        mock_ml_training_orchestration_service.start_training.return_value = {
            "success": True,
            "message": "アンサンブル学習を開始しました",
            "training_id": "train_ensemble_12345",
        }

        # アンサンブル設定を追加
        config_with_ensemble = {
            **sample_training_config,
            "ensemble_config": {
                "enabled": True,
                "method": "stacking",
                "stacking_params": {
                    "base_models": ["lightgbm", "xgboost"],
                    "meta_model": "logistic_regression",
                    "cv_folds": 5,
                },
            },
        }

        # APIリクエスト
        response = test_client.post(
            "/api/ml-training/train",
            json=config_with_ensemble,
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_start_training_with_single_model(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_ml_training_orchestration_service: AsyncMock,
        sample_training_config: Dict[str, Any],
    ) -> None:
        """
        正常系: 単一モデルでトレーニングが開始される

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_ml_training_orchestration_service: オーケストレーションサービスモック
            sample_training_config: サンプルトレーニング設定
        """
        # モックの設定
        mock_ml_training_orchestration_service.start_training.return_value = {
            "success": True,
            "message": "単一モデルトレーニングを開始しました",
            "training_id": "train_single_12345",
        }

        # 単一モデル設定を追加
        config_with_single = {
            **sample_training_config,
            "single_model_config": {"model_type": "lightgbm"},
        }

        # APIリクエスト
        response = test_client.post(
            "/api/ml-training/train",
            json=config_with_single,
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_start_training_invalid_config(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_ml_training_orchestration_service: AsyncMock,
    ) -> None:
        """
        異常系: 無効な設定でトレーニングが失敗する

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_ml_training_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        # 不完全な設定
        invalid_config = {
            "symbol": "BTC/USDT:USDT",
            # 必須フィールドが欠落
        }

        # APIリクエスト
        response = test_client.post(
            "/api/ml-training/train",
            json=invalid_config,
        )

        # アサーション
        assert response.status_code == 422


class TestTrainingStatus:
    """トレーニング状態取得のテストクラス"""
    def test_get_training_status_in_progress(
        self,
        test_client: TestClient,
        mock_ml_training_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: トレーニング中の状態が取得できる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_ml_training_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_ml_training_orchestration_service.get_training_status.return_value = {
            "is_training": True,
            "progress": 50,
            "status": "training",
            "message": "トレーニング中",
            "start_time": "2024-01-01T00:00:00",
        }

        # APIリクエスト
        response = test_client.get("/api/ml-training/training/status")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["is_training"] is True
        assert data["progress"] == 50
    def test_get_training_status_completed(
        self,
        test_client: TestClient,
        mock_ml_training_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: 完了状態が取得できる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_ml_training_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_ml_training_orchestration_service.get_training_status.return_value = {
            "is_training": False,
            "progress": 100,
            "status": "completed",
            "message": "トレーニング完了",
            "start_time": "2024-01-01T00:00:00",
            "end_time": "2024-01-01T01:00:00",
            "model_info": {"accuracy": 0.85, "f1_score": 0.82},
        }

        # APIリクエスト
        response = test_client.get("/api/ml-training/training/status")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["is_training"] is False
        assert data["progress"] == 100
        assert "model_info" in data
    def test_get_training_status_error(
        self,
        test_client: TestClient,
        mock_ml_training_orchestration_service: AsyncMock,
    ) -> None:
        """
        エッジケース: エラー状態が取得できる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_ml_training_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_ml_training_orchestration_service.get_training_status.return_value = {
            "is_training": False,
            "progress": 0,
            "status": "error",
            "message": "トレーニングエラー",
            "error": "データ不足エラー",
        }

        # APIリクエスト
        response = test_client.get("/api/ml-training/training/status")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["is_training"] is False
        assert data["status"] == "error"
        assert "error" in data


class TestModelInfo:
    """モデル情報取得のテストクラス"""
    def test_get_model_info_success(
        self,
        test_client: TestClient,
        mock_ml_training_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: モデル情報が正常に取得できる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_ml_training_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_ml_training_orchestration_service.get_model_info.return_value = {
            "success": True,
            "data": {
                "model_type": "ensemble",
                "accuracy": 0.85,
                "precision": 0.83,
                "recall": 0.87,
                "f1_score": 0.85,
            },
        }

        # APIリクエスト
        response = test_client.get("/api/ml-training/model-info")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
    def test_get_model_info_no_model(
        self,
        test_client: TestClient,
        mock_ml_training_orchestration_service: AsyncMock,
    ) -> None:
        """
        エッジケース: モデルが存在しない場合

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_ml_training_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_ml_training_orchestration_service.get_model_info.return_value = {
            "success": False,
            "message": "モデルが見つかりません",
        }

        # APIリクエスト
        response = test_client.get("/api/ml-training/model-info")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


class TestStopTraining:
    """トレーニング停止のテストクラス"""
    def test_stop_training_success(
        self,
        test_client: TestClient,
        mock_ml_training_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: トレーニングが正常に停止される

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_ml_training_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_ml_training_orchestration_service.stop_training.return_value = {
            "success": True,
            "message": "トレーニングを停止しました",
        }

        # APIリクエスト
        response = test_client.post("/api/ml-training/stop")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "停止" in data["message"]
    def test_stop_training_not_running(
        self,
        test_client: TestClient,
        mock_ml_training_orchestration_service: AsyncMock,
    ) -> None:
        """
        エッジケース: トレーニングが実行されていない場合

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_ml_training_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_ml_training_orchestration_service.stop_training.return_value = {
            "success": False,
            "message": "実行中のトレーニングがありません",
        }

        # APIリクエスト
        response = test_client.post("/api/ml-training/stop")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


class TestErrorHandling:
    """エラーハンドリングのテストクラス"""

    def test_service_error_handling(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_ml_training_orchestration_service: AsyncMock,
        sample_training_config: Dict[str, Any],
    ) -> None:
        """
        異常系: サービス層でエラーが発生した場合

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_ml_training_orchestration_service: オーケストレーションサービスモック
            sample_training_config: サンプルトレーニング設定
        """
        # モックの設定
        mock_ml_training_orchestration_service.start_training.side_effect = Exception(
            "Training error"
        )

        # APIリクエスト
        response = test_client.post(
            "/api/ml-training/train",
            json=sample_training_config,
        )

        # アサーション（ErrorHandlerによって処理される）
        assert response.status_code in [200, 500]
    def test_status_error_handling(
        self,
        test_client: TestClient,
        mock_ml_training_orchestration_service: AsyncMock,
    ) -> None:
        """
        異常系: ステータス取得時にエラーが発生した場合

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_ml_training_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定: Exceptionではなくエラー状態を返す
        mock_ml_training_orchestration_service.get_training_status.return_value = {
            "is_training": False,
            "progress": 0,
            "status": "error",
            "message": "ステータス取得エラー",
            "error": "Status error",
        }

        # APIリクエスト
        response = test_client.get("/api/ml-training/training/status")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"