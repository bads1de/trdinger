"""
バックテストAPIのテストモジュール

バックテストAPIエンドポイントの正常系、異常系、エッジケースをテストします。
"""

from typing import Any, Dict
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_backtest_orchestration_service, get_db
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
def mock_backtest_orchestration_service() -> AsyncMock:
    """
    BacktestOrchestrationServiceのモック

    Returns:
        AsyncMock: モックされたバックテストオーケストレーションサービス
    """
    mock_service = AsyncMock()
    mock_service.get_backtest_results = AsyncMock()
    mock_service.get_backtest_result_by_id = AsyncMock()
    mock_service.delete_backtest_result = AsyncMock()
    mock_service.delete_all_backtest_results = AsyncMock()
    mock_service.get_supported_strategies = AsyncMock()
    return mock_service


@pytest.fixture(autouse=True)
def override_dependencies(mock_db_session, mock_backtest_orchestration_service):
    """
    FastAPIの依存性注入をオーバーライド

    Args:
        mock_db_session: モックDBセッション
        mock_backtest_orchestration_service: モックサービス
    """
    app.dependency_overrides[get_db] = lambda: mock_db_session
    app.dependency_overrides[get_backtest_orchestration_service] = (
        lambda: mock_backtest_orchestration_service
    )
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def sample_backtest_result() -> Dict[str, Any]:
    """
    サンプルバックテスト結果

    Returns:
        Dict[str, Any]: バックテスト結果のサンプルデータ
    """
    return {
        "id": 1,
        "strategy_name": "test_strategy",
        "symbol": "BTC/USDT:USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01T00:00:00",
        "end_date": "2024-01-31T00:00:00",
        "initial_capital": 10000.0,
        "final_capital": 11000.0,
        "total_return": 0.1,
        "sharpe_ratio": 1.5,
        "max_drawdown": 0.05,
        "total_trades": 10,
        "winning_trades": 6,
        "losing_trades": 4,
    }


@pytest.fixture
def sample_backtest_results_list(sample_backtest_result: Dict[str, Any]) -> list:
    """
    サンプルバックテスト結果リスト

    Args:
        sample_backtest_result: 単一のバックテスト結果

    Returns:
        list: バックテスト結果のリスト
    """
    return [
        sample_backtest_result,
        {**sample_backtest_result, "id": 2, "strategy_name": "test_strategy_2"},
    ]


class TestBacktestResultsRetrieval:
    """バックテスト結果取得のテストクラス"""

    def test_get_backtest_results_success(
        self,
        test_client: TestClient,
        mock_backtest_orchestration_service: AsyncMock,
        sample_backtest_results_list: list,
    ) -> None:
        """
        正常系: バックテスト結果一覧が正常に取得できる

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_backtest_orchestration_service: オーケストレーションサービスモック
            sample_backtest_results_list: サンプル結果リスト
        """
        # モックの設定
        mock_backtest_orchestration_service.get_backtest_results.return_value = {
            "success": True,
            "results": sample_backtest_results_list,
            "total": 2,
        }

        # APIリクエスト
        response = test_client.get("/api/backtest/results")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["results"]) == 2
        assert data["total"] == 2
        assert data["results"][0]["id"] == 1
        assert data["results"][1]["id"] == 2

    def test_get_backtest_results_with_filters(
        self,
        test_client: TestClient,
        mock_backtest_orchestration_service: AsyncMock,
        sample_backtest_result: Dict[str, Any],
    ) -> None:
        """
        正常系: フィルター付きでバックテスト結果が取得できる

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_backtest_orchestration_service: オーケストレーションサービスモック
            sample_backtest_result: サンプル結果
        """
        # モックの設定
        mock_backtest_orchestration_service.get_backtest_results.return_value = {
            "success": True,
            "results": [sample_backtest_result],
            "total": 1,
        }

        # APIリクエスト（フィルター付き）
        response = test_client.get(
            "/api/backtest/results",
            params={
                "limit": 10,
                "offset": 0,
                "symbol": "BTC/USDT:USDT",
                "strategy_name": "test_strategy",
            },
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["results"]) == 1
        assert data["results"][0]["symbol"] == "BTC/USDT:USDT"

    def test_get_backtest_results_empty(
        self,
        test_client: TestClient,
        mock_backtest_orchestration_service: AsyncMock,
    ) -> None:
        """
        エッジケース: 結果が空の場合

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_backtest_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_backtest_orchestration_service.get_backtest_results.return_value = {
            "success": True,
            "results": [],
            "total": 0,
        }

        # APIリクエスト
        response = test_client.get("/api/backtest/results")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["results"]) == 0
        assert data["total"] == 0

    @pytest.mark.parametrize(
        "limit,offset,expected_status",
        [
            (1, 0, 200),  # 最小値
            (50, 100, 200),  # 通常値
            (0, 0, 422),  # 無効な最小値
            (-1, 0, 422),  # 負の値
            (0, -1, 422),  # 負のオフセット
        ],
    )
    def test_get_backtest_results_validation(
        self,
        test_client: TestClient,
        mock_backtest_orchestration_service: AsyncMock,
        limit: int,
        offset: int,
        expected_status: int,
    ) -> None:
        """
        異常系: パラメータバリデーションのテスト

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_backtest_orchestration_service: オーケストレーションサービスモック
            limit: 取得件数
            offset: オフセット
            expected_status: 期待されるステータスコード
        """
        # モックの設定
        mock_backtest_orchestration_service.get_backtest_results.return_value = {
            "success": True,
            "results": [],
            "total": 0,
        }

        # APIリクエスト
        response = test_client.get(
            "/api/backtest/results", params={"limit": limit, "offset": offset}
        )

        # アサーション
        assert response.status_code == expected_status


class TestBacktestResultById:
    """ID指定でのバックテスト結果取得のテストクラス"""

    def test_get_backtest_result_by_id_success(
        self,
        test_client: TestClient,
        mock_backtest_orchestration_service: AsyncMock,
        sample_backtest_result: Dict[str, Any],
    ) -> None:
        """
        正常系: ID指定でバックテスト結果が取得できる

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_backtest_orchestration_service: オーケストレーションサービスモック
            sample_backtest_result: サンプル結果
        """
        # モックの設定
        mock_backtest_orchestration_service.get_backtest_result_by_id.return_value = {
            "success": True,
            "result": sample_backtest_result,
        }

        # APIリクエスト
        response = test_client.get("/api/backtest/results/1/")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        # api_response形式: トップレベルにresultキーがある
        assert data["result"]["id"] == 1
        assert data["result"]["strategy_name"] == "test_strategy"

    def test_get_backtest_result_by_id_not_found(
        self,
        test_client: TestClient,
        mock_backtest_orchestration_service: AsyncMock,
    ) -> None:
        """
        異常系: 存在しないIDで404エラーが返される

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_backtest_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_backtest_orchestration_service.get_backtest_result_by_id.return_value = {
            "success": False,
            "error": "Backtest result not found",
            "status_code": 404,
        }

        # APIリクエスト
        response = test_client.get("/api/backtest/results/9999/")

        # アサーション
        assert response.status_code == 404


class TestBacktestResultDeletion:
    """バックテスト結果削除のテストクラス"""

    def test_delete_backtest_result_success(
        self,
        test_client: TestClient,
        mock_backtest_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: バックテスト結果が正常に削除できる

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_backtest_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_backtest_orchestration_service.delete_backtest_result.return_value = {
            "success": True,
            "message": "バックテスト結果を削除しました",
        }

        # APIリクエスト
        response = test_client.delete("/api/backtest/results/1/")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "削除" in data["message"]

    def test_delete_backtest_result_not_found(
        self,
        test_client: TestClient,
        mock_backtest_orchestration_service: AsyncMock,
    ) -> None:
        """
        異常系: 存在しないIDの削除で404エラーが返される

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_backtest_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_backtest_orchestration_service.delete_backtest_result.return_value = {
            "success": False,
            "error": "Backtest result not found",
            "status_code": 404,
        }

        # APIリクエスト
        response = test_client.delete("/api/backtest/results/9999/")

        # アサーション
        assert response.status_code == 404

    def test_delete_all_backtest_results_success(
        self,
        test_client: TestClient,
        mock_backtest_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: すべてのバックテスト結果が正常に削除できる

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_backtest_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_backtest_orchestration_service.delete_all_backtest_results.return_value = {
            "success": True,
            "message": "すべてのバックテスト関連データを削除しました",
            "data": {
                "deleted_backtest_results": 10,
                "deleted_ga_experiments": 5,
                "deleted_generated_strategies": 8,
            },
        }

        # APIリクエスト
        response = test_client.delete("/api/backtest/results-all")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "削除" in data["message"]
        assert data["data"]["deleted_backtest_results"] == 10


class TestSupportedStrategies:
    """サポート戦略取得のテストクラス"""

    def test_get_supported_strategies_success(
        self,
        test_client: TestClient,
        mock_backtest_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: サポート戦略一覧が正常に取得できる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_backtest_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_backtest_orchestration_service.get_supported_strategies.return_value = {
            "success": True,
            "data": {
                "strategies": [
                    "sma_crossover",
                    "rsi_strategy",
                    "bollinger_bands",
                ]
            },
        }

        # APIリクエスト
        response = test_client.get("/api/backtest/strategies")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["strategies"]) == 3
        assert "sma_crossover" in data["data"]["strategies"]

    def test_get_supported_strategies_empty(
        self,
        test_client: TestClient,
        mock_backtest_orchestration_service: AsyncMock,
    ) -> None:
        """
        エッジケース: 戦略が空の場合

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_backtest_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_backtest_orchestration_service.get_supported_strategies.return_value = {
            "success": True,
            "data": {"strategies": []},
        }

        # APIリクエスト
        response = test_client.get("/api/backtest/strategies")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["strategies"]) == 0


class TestErrorHandling:
    """エラーハンドリングのテストクラス"""

    def test_service_error_handling(
        self,
        test_client: TestClient,
        mock_backtest_orchestration_service: AsyncMock,
    ) -> None:
        """
        異常系: サービス層でエラーが発生した場合

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_backtest_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_backtest_orchestration_service.get_backtest_results.return_value = {
            "success": False,
            "error": "Database connection error",
            "status_code": 500,
        }

        # APIリクエスト
        response = test_client.get("/api/backtest/results")

        # アサーション
        assert response.status_code == 200  # ErrorHandlerによりラップされる
        data = response.json()
        assert data["success"] is False
        assert "error" in data

    def test_unexpected_exception_handling(
        self,
        test_client: TestClient,
        mock_backtest_orchestration_service: AsyncMock,
    ) -> None:
        """
        異常系: 予期しない例外が発生した場合

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_backtest_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_backtest_orchestration_service.get_backtest_results.side_effect = (
            Exception("Unexpected error")
        )

        # APIリクエスト
        response = test_client.get("/api/backtest/results")

        # アサーション（ErrorHandlerによって処理される）
        assert response.status_code in [200, 500]
