"""
AutoStrategy APIのテストモジュール

AutoStrategy APIエンドポイントの正常系、異常系、エッジケースをテストします。
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_auto_strategy_service, get_db
from app.main import app


@pytest.fixture
def test_client() -> TestClient:
    """
    FastAPIテストクライアントのフィクスチャ

    Returns:
        TestClient: テスト用のFastAPIクライアント
    """
    return TestClient(app)


@pytest.fixture
def mock_db_session() -> Mock:
    """
    データベースセッションのモック

    Returns:
        Mock: モックされたデータベースセッション
    """
    return Mock()


@pytest.fixture
def mock_auto_strategy_service() -> AsyncMock:
    """
    AutoStrategyServiceのモック

    Returns:
        AsyncMock: モックされたAutoStrategyサービス
    """
    mock_service = AsyncMock()
    mock_service.start_strategy_generation = Mock()
    mock_service.list_experiments = Mock()
    mock_service.stop_experiment = Mock()
    return mock_service


@pytest.fixture(autouse=True)
def override_dependencies(mock_db_session, mock_auto_strategy_service):
    """
    FastAPIの依存性注入をオーバーライド

    Args:
        mock_db_session: モックDBセッション
        mock_auto_strategy_service: モックAutoStrategyサービス
    """
    app.dependency_overrides[get_db] = lambda: mock_db_session
    app.dependency_overrides[get_auto_strategy_service] = (
        lambda: mock_auto_strategy_service
    )
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def sample_ga_generation_request() -> Dict[str, Any]:
    """
    サンプルGA戦略生成リクエスト

    Returns:
        Dict[str, Any]: GA生成リクエストのサンプルデータ
    """
    return {
        "experiment_id": "test-exp-001",
        "experiment_name": "BTC_Strategy_Gen_001",
        "base_config": {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-12-19",
            "initial_capital": 100000,
            "commission_rate": 0.00055,
        },
        "ga_config": {
            "population_size": 10,
            "generations": 5,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "elite_size": 2,
            "max_indicators": 3,
            "enable_multi_objective": False,
            "objectives": ["total_return"],
            "objective_weights": [1.0],
        },
    }


@pytest.fixture
def sample_experiment() -> Dict[str, Any]:
    """
    サンプル実験データ

    Returns:
        Dict[str, Any]: 実験データのサンプル
    """
    return {
        "experiment_id": "test-exp-001",
        "experiment_name": "BTC_Strategy_Gen_001",
        "status": "running",
        "progress": 0.5,
        "current_generation": 3,
        "total_generations": 5,
        "best_fitness": 0.85,
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T01:00:00",
    }


@pytest.fixture
def sample_experiments_list(sample_experiment: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    サンプル実験リスト

    Args:
        sample_experiment: 単一の実験データ

    Returns:
        List[Dict[str, Any]]: 実験データのリスト
    """
    return [
        sample_experiment,
        {
            **sample_experiment,
            "experiment_id": "test-exp-002",
            "status": "completed",
            "progress": 1.0,
        },
        {
            **sample_experiment,
            "experiment_id": "test-exp-003",
            "status": "failed",
            "progress": 0.2,
        },
    ]


class TestGenerateStrategy:
    """GA戦略生成のテストクラス"""

    def test_generate_strategy_success(
        self,
        test_client: TestClient,
        mock_auto_strategy_service: AsyncMock,
        sample_ga_generation_request: Dict[str, Any],
    ) -> None:
        """
        正常系: GA戦略生成が正常に開始される

        Args:
            test_client: テストクライアント
            mock_auto_strategy_service: AutoStrategyサービスモック
            sample_ga_generation_request: サンプル生成リクエスト
        """
        # モックの設定
        mock_auto_strategy_service.start_strategy_generation.return_value = (
            "test-exp-001"
        )

        # APIリクエスト
        response = test_client.post(
            "/api/auto-strategy/generate", json=sample_ga_generation_request
        )

        # アサーション
        assert response.status_code == 202
        data = response.json()
        assert data["success"] is True
        assert "experiment_id" in data["data"]
        assert data["data"]["experiment_id"] == "test-exp-001"
        assert "開始" in data["message"]

    def test_generate_strategy_with_multi_objective(
        self,
        test_client: TestClient,
        mock_auto_strategy_service: AsyncMock,
        sample_ga_generation_request: Dict[str, Any],
    ) -> None:
        """
        正常系: 多目的最適化でのGA戦略生成

        Args:
            test_client: テストクライアント
            mock_auto_strategy_service: AutoStrategyサービスモック
            sample_ga_generation_request: サンプル生成リクエスト
        """
        # 多目的最適化設定を追加
        sample_ga_generation_request["ga_config"]["enable_multi_objective"] = True
        sample_ga_generation_request["ga_config"]["objectives"] = [
            "total_return",
            "sharpe_ratio",
        ]
        sample_ga_generation_request["ga_config"]["objective_weights"] = [0.6, 0.4]

        # モックの設定
        mock_auto_strategy_service.start_strategy_generation.return_value = (
            "test-exp-multi-001"
        )

        # APIリクエスト
        response = test_client.post(
            "/api/auto-strategy/generate", json=sample_ga_generation_request
        )

        # アサーション
        assert response.status_code == 202
        data = response.json()
        assert data["success"] is True
        assert data["data"]["experiment_id"] == "test-exp-multi-001"

    def test_generate_strategy_minimal_config(
        self,
        test_client: TestClient,
        mock_auto_strategy_service: AsyncMock,
    ) -> None:
        """
        正常系: 最小限の設定でのGA戦略生成

        Args:
            test_client: テストクライアント
            mock_auto_strategy_service: AutoStrategyサービスモック
        """
        # 最小限の設定
        minimal_request = {
            "experiment_id": "test-exp-minimal",
            "experiment_name": "Minimal_Test",
            "base_config": {
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
            },
            "ga_config": {
                "population_size": 10,
                "generations": 5,
            },
        }

        # モックの設定
        mock_auto_strategy_service.start_strategy_generation.return_value = (
            "test-exp-minimal"
        )

        # APIリクエスト
        response = test_client.post("/api/auto-strategy/generate", json=minimal_request)

        # アサーション
        assert response.status_code == 202
        data = response.json()
        assert data["success"] is True

    @pytest.mark.parametrize(
        "missing_field,expected_status",
        [
            ("experiment_id", 422),
            ("experiment_name", 422),
            ("base_config", 422),
            ("ga_config", 422),
        ],
    )
    def test_generate_strategy_missing_fields(
        self,
        test_client: TestClient,
        sample_ga_generation_request: Dict[str, Any],
        missing_field: str,
        expected_status: int,
    ) -> None:
        """
        異常系: 必須フィールド欠損時のバリデーション

        Args:
            test_client: テストクライアント
            sample_ga_generation_request: サンプル生成リクエスト
            missing_field: 欠損させるフィールド名
            expected_status: 期待されるステータスコード
        """
        # フィールドを削除
        invalid_request = sample_ga_generation_request.copy()
        del invalid_request[missing_field]

        # APIリクエスト
        response = test_client.post("/api/auto-strategy/generate", json=invalid_request)

        # アサーション
        assert response.status_code == expected_status

    def test_generate_strategy_service_error(
        self,
        test_client: TestClient,
        mock_auto_strategy_service: AsyncMock,
        sample_ga_generation_request: Dict[str, Any],
    ) -> None:
        """
        異常系: サービス層でエラーが発生した場合

        Args:
            test_client: テストクライアント
            mock_auto_strategy_service: AutoStrategyサービスモック
            sample_ga_generation_request: サンプル生成リクエスト
        """
        # モックの設定
        mock_auto_strategy_service.start_strategy_generation.side_effect = Exception(
            "GA engine initialization failed"
        )

        # APIリクエスト
        response = test_client.post(
            "/api/auto-strategy/generate", json=sample_ga_generation_request
        )

        # アサーション（エラーハンドラで202とerror_responseが返される）
        assert response.status_code == 202
        data = response.json()
        assert data["success"] is False
        assert "error" in data or "GA engine initialization failed" in data["message"]


class TestListExperiments:
    """実験一覧取得のテストクラス"""

    def test_list_experiments_success(
        self,
        test_client: TestClient,
        mock_auto_strategy_service: AsyncMock,
        sample_experiments_list: List[Dict[str, Any]],
    ) -> None:
        """
        正常系: 実験一覧が正常に取得できる

        Args:
            test_client: テストクライアント
            mock_auto_strategy_service: AutoStrategyサービスモック
            sample_experiments_list: サンプル実験リスト
        """
        # モックの設定
        mock_auto_strategy_service.list_experiments.return_value = (
            sample_experiments_list
        )

        # APIリクエスト
        response = test_client.get("/api/auto-strategy/experiments")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert "experiments" in data
        assert len(data["experiments"]) == 3
        assert data["experiments"][0]["experiment_id"] == "test-exp-001"

    def test_list_experiments_empty(
        self,
        test_client: TestClient,
        mock_auto_strategy_service: AsyncMock,
    ) -> None:
        """
        エッジケース: 実験が存在しない場合

        Args:
            test_client: テストクライアント
            mock_auto_strategy_service: AutoStrategyサービスモック
        """
        # モックの設定
        mock_auto_strategy_service.list_experiments.return_value = []

        # APIリクエスト
        response = test_client.get("/api/auto-strategy/experiments")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert "experiments" in data
        assert len(data["experiments"]) == 0

    def test_list_experiments_with_various_statuses(
        self,
        test_client: TestClient,
        mock_auto_strategy_service: AsyncMock,
        sample_experiments_list: List[Dict[str, Any]],
    ) -> None:
        """
        正常系: 様々なステータスの実験が取得できる

        Args:
            test_client: テストクライアント
            mock_auto_strategy_service: AutoStrategyサービスモック
            sample_experiments_list: サンプル実験リスト
        """
        # モックの設定
        mock_auto_strategy_service.list_experiments.return_value = (
            sample_experiments_list
        )

        # APIリクエスト
        response = test_client.get("/api/auto-strategy/experiments")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        statuses = [exp["status"] for exp in data["experiments"]]
        assert "running" in statuses
        assert "completed" in statuses
        assert "failed" in statuses

    def test_list_experiments_service_error(
        self,
        test_client: TestClient,
        mock_auto_strategy_service: AsyncMock,
    ) -> None:
        """
        異常系: サービス層でエラーが発生した場合

        Args:
            test_client: テストクライアント
            mock_auto_strategy_service: AutoStrategyサービスモック
        """
        # モックの設定
        mock_auto_strategy_service.list_experiments.side_effect = Exception(
            "Database connection error"
        )

        # APIリクエスト
        response = test_client.get("/api/auto-strategy/experiments")

        # アサーション（ErrorHandlerによって処理される）
        assert response.status_code in [200, 500]


class TestStopExperiment:
    """実験停止のテストクラス"""

    def test_stop_experiment_success(
        self,
        test_client: TestClient,
        mock_auto_strategy_service: AsyncMock,
    ) -> None:
        """
        正常系: 実験が正常に停止される

        Args:
            test_client: テストクライアント
            mock_auto_strategy_service: AutoStrategyサービスモック
        """
        # モックの設定
        mock_auto_strategy_service.stop_experiment.return_value = {
            "success": True,
            "message": "実験を停止しました",
        }

        # APIリクエスト
        response = test_client.post("/api/auto-strategy/experiments/test-exp-001/stop")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "停止" in data["message"]

    def test_stop_experiment_not_found(
        self,
        test_client: TestClient,
        mock_auto_strategy_service: AsyncMock,
    ) -> None:
        """
        異常系: 存在しない実験を停止しようとした場合

        Args:
            test_client: テストクライアント
            mock_auto_strategy_service: AutoStrategyサービスモック
        """
        # モックの設定
        mock_auto_strategy_service.stop_experiment.side_effect = ValueError(
            "Experiment not found"
        )

        # APIリクエスト
        response = test_client.post(
            "/api/auto-strategy/experiments/nonexistent-exp/stop"
        )

        # アサーション
        assert response.status_code == 400

    def test_stop_experiment_already_stopped(
        self,
        test_client: TestClient,
        mock_auto_strategy_service: AsyncMock,
    ) -> None:
        """
        異常系: 既に停止済みの実験を停止しようとした場合

        Args:
            test_client: テストクライアント
            mock_auto_strategy_service: AutoStrategyサービスモック
        """
        # モックの設定
        mock_auto_strategy_service.stop_experiment.side_effect = ValueError(
            "Experiment already stopped"
        )

        # APIリクエスト
        response = test_client.post("/api/auto-strategy/experiments/test-exp-001/stop")

        # アサーション
        assert response.status_code == 400

    def test_stop_experiment_completed(
        self,
        test_client: TestClient,
        mock_auto_strategy_service: AsyncMock,
    ) -> None:
        """
        エッジケース: 完了済みの実験を停止しようとした場合

        Args:
            test_client: テストクライアント
            mock_auto_strategy_service: AutoStrategyサービスモック
        """
        # モックの設定
        mock_auto_strategy_service.stop_experiment.return_value = {
            "success": False,
            "message": "実験は既に完了しています",
        }

        # APIリクエスト
        response = test_client.post("/api/auto-strategy/experiments/test-exp-001/stop")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


class TestErrorHandling:
    """エラーハンドリングのテストクラス"""

    def test_invalid_json_format(
        self,
        test_client: TestClient,
    ) -> None:
        """
        異常系: 無効なJSON形式のリクエスト

        Args:
            test_client: テストクライアント
        """
        # APIリクエスト（無効なJSON）
        response = test_client.post(
            "/api/auto-strategy/generate",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )

        # アサーション
        assert response.status_code == 422

    def test_empty_request_body(
        self,
        test_client: TestClient,
    ) -> None:
        """
        異常系: 空のリクエストボディ

        Args:
            test_client: テストクライアント
        """
        # APIリクエスト（空のボディ）
        response = test_client.post("/api/auto-strategy/generate", json={})

        # アサーション
        assert response.status_code == 422

    def test_unexpected_exception_handling(
        self,
        test_client: TestClient,
        mock_auto_strategy_service: AsyncMock,
        sample_ga_generation_request: Dict[str, Any],
    ) -> None:
        """
        異常系: 予期しない例外が発生した場合

        Args:
            test_client: テストクライアント
            mock_auto_strategy_service: AutoStrategyサービスモック
            sample_ga_generation_request: サンプル生成リクエスト
        """
        # モックの設定
        mock_auto_strategy_service.start_strategy_generation.side_effect = Exception(
            "Unexpected error"
        )

        # APIリクエスト
        response = test_client.post(
            "/api/auto-strategy/generate", json=sample_ga_generation_request
        )

        # アサーション（エラーハンドラで202とerror_responseが返される）
        assert response.status_code == 202
        data = response.json()
        assert data["success"] is False
        assert "error" in data or "Unexpected error" in data["message"]


class TestBackgroundTaskExecution:
    """バックグラウンドタスク実行のテストクラス"""

    def test_generate_strategy_returns_immediately(
        self,
        test_client: TestClient,
        mock_auto_strategy_service: AsyncMock,
        sample_ga_generation_request: Dict[str, Any],
    ) -> None:
        """
        正常系: GA戦略生成が即座にレスポンスを返す

        Args:
            test_client: テストクライアント
            mock_auto_strategy_service: AutoStrategyサービスモック
            sample_ga_generation_request: サンプル生成リクエスト
        """
        # モックの設定
        mock_auto_strategy_service.start_strategy_generation.return_value = (
            "test-exp-001"
        )

        # APIリクエスト
        response = test_client.post(
            "/api/auto-strategy/generate", json=sample_ga_generation_request
        )

        # アサーション
        assert response.status_code == 202  # Accepted
        data = response.json()
        assert data["success"] is True
        # バックグラウンドタスクが開始されたことを確認
        mock_auto_strategy_service.start_strategy_generation.assert_called_once()

    def test_multiple_concurrent_requests(
        self,
        test_client: TestClient,
        mock_auto_strategy_service: AsyncMock,
        sample_ga_generation_request: Dict[str, Any],
    ) -> None:
        """
        正常系: 複数の並行リクエストが処理される

        Args:
            test_client: テストクライアント
            mock_auto_strategy_service: AutoStrategyサービスモック
            sample_ga_generation_request: サンプル生成リクエスト
        """
        # モックの設定
        mock_auto_strategy_service.start_strategy_generation.side_effect = [
            "test-exp-001",
            "test-exp-002",
            "test-exp-003",
        ]

        # 複数のAPIリクエスト
        responses = []
        for i in range(3):
            request_data = sample_ga_generation_request.copy()
            request_data["experiment_id"] = f"test-exp-00{i+1}"
            response = test_client.post(
                "/api/auto-strategy/generate", json=request_data
            )
            responses.append(response)

        # アサーション
        for response in responses:
            assert response.status_code == 202
            assert response.json()["success"] is True
