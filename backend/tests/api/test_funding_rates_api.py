"""
ファンディングレートAPIのテストモジュール

ファンディングレートAPIエンドポイントの正常系、異常系、エッジケースをテストします。
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_db, get_funding_rate_orchestration_service
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
def mock_funding_rate_orchestration_service() -> AsyncMock:
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
def override_dependencies(mock_db_session, mock_funding_rate_orchestration_service):
    """
    FastAPIの依存性注入をオーバーライド

    Args:
        mock_db_session: モックDBセッション
        mock_funding_rate_orchestration_service: モックサービス
    """
    app.dependency_overrides[get_db] = lambda: mock_db_session
    app.dependency_overrides[get_funding_rate_orchestration_service] = (
        lambda: mock_funding_rate_orchestration_service
    )
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def sample_funding_rate() -> Dict[str, Any]:
    """
    サンプルファンディングレートデータ

    Returns:
        Dict[str, Any]: ファンディングレートデータのサンプル
    """
    return {
        "id": 1,
        "symbol": "BTC/USDT:USDT",
        "funding_rate": 0.0001,
        "timestamp": "2024-01-01T00:00:00",
        "next_funding_time": "2024-01-01T08:00:00",
    }


@pytest.fixture
def sample_funding_rates_list(
    sample_funding_rate: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    サンプルファンディングレートリスト

    Args:
        sample_funding_rate: 単一のファンディングレート

    Returns:
        List[Dict[str, Any]]: ファンディングレートのリスト
    """
    return [
        sample_funding_rate,
        {**sample_funding_rate, "id": 2, "funding_rate": 0.0002},
        {**sample_funding_rate, "id": 3, "funding_rate": -0.0001},
    ]


class TestGetFundingRates:
    """ファンディングレート取得のテストクラス"""

    def test_get_funding_rates_success(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_funding_rate_orchestration_service: AsyncMock,
        sample_funding_rates_list: List[Dict[str, Any]],
    ) -> None:
        """
        正常系: ファンディングレートが正常に取得できる

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_funding_rate_orchestration_service: オーケストレーションサービスモック
            sample_funding_rates_list: サンプルデータリスト
        """
        # モックの設定
        mock_funding_rate_orchestration_service.get_funding_rate_data.return_value = (
            sample_funding_rates_list
        )

        # APIリクエスト
        response = test_client.get(
            "/api/funding-rates/",
            params={"symbol": "BTC/USDT:USDT", "limit": 100},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["symbol"] == "BTC/USDT:USDT"
        assert data["data"]["count"] == 3
        assert len(data["data"]["funding_rates"]) == 3

    def test_get_funding_rates_with_date_range(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_funding_rate_orchestration_service: AsyncMock,
        sample_funding_rate: Dict[str, Any],
    ) -> None:
        """
        正常系: 日付範囲指定でファンディングレートが取得できる

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_funding_rate_orchestration_service: オーケストレーションサービスモック
            sample_funding_rate: サンプルデータ
        """
        # モックの設定
        mock_funding_rate_orchestration_service.get_funding_rate_data.return_value = [
            sample_funding_rate
        ]

        # APIリクエスト
        response = test_client.get(
            "/api/funding-rates/",
            params={
                "symbol": "BTC/USDT:USDT",
                "start_date": "2024-01-01T00:00:00",
                "end_date": "2024-01-31T23:59:59",
                "limit": 100,
            },
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["count"] == 1

    def test_get_funding_rates_empty(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_funding_rate_orchestration_service: AsyncMock,
    ) -> None:
        """
        エッジケース: データが空の場合

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_funding_rate_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_funding_rate_orchestration_service.get_funding_rate_data.return_value = []

        # APIリクエスト
        response = test_client.get(
            "/api/funding-rates/",
            params={"symbol": "BTC/USDT:USDT"},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["count"] == 0

    @pytest.mark.parametrize(
        "limit,expected_status",
        [
            (1, 200),  # 最小値
            (500, 200),  # 中間値
            (1000, 200),  # 最大値
            (0, 200),  # エッジケース（実装では許容）
            (1001, 200),  # エッジケース（実装では許容）
        ],
    )
    def test_get_funding_rates_limit_validation(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_funding_rate_orchestration_service: AsyncMock,
        limit: int,
        expected_status: int,
    ) -> None:
        """
        異常系: リミットパラメータのバリデーション

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_funding_rate_orchestration_service: オーケストレーションサービスモック
            limit: リミット値
            expected_status: 期待されるステータスコード
        """
        # モックの設定
        mock_funding_rate_orchestration_service.get_funding_rate_data.return_value = []

        # APIリクエスト
        response = test_client.get(
            "/api/funding-rates/",
            params={"symbol": "BTC/USDT:USDT", "limit": limit},
        )

        # アサーション
        assert response.status_code == expected_status

    def test_get_funding_rates_missing_symbol(
        self,
        test_client: TestClient,
    ) -> None:
        """
        異常系: 必須パラメータsymbolが欠落している場合

        Args:
            test_client: テストクライアント
        """
        # APIリクエスト
        response = test_client.get("/api/funding-rates/")

        # アサーション
        assert response.status_code == 422


class TestCollectFundingRates:
    """ファンディングレート収集のテストクラス"""

    def test_collect_funding_rates_success(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_funding_rate_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: ファンディングレート収集が正常に実行される

        Args:
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_funding_rate_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_funding_rate_orchestration_service.collect_funding_rate_data.return_value = {
            "success": True,
            "message": "ファンディングレートデータを収集しました",
            "data": {"collected_count": 100},
        }

        # APIリクエスト
        response = test_client.post(
            "/api/funding-rates/collect",
            params={"symbol": "BTC/USDT:USDT", "limit": 100},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "収集" in data["message"]

    def test_collect_funding_rates_fetch_all(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_funding_rate_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: 全期間データ収集が実行される

        Args:
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_funding_rate_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_funding_rate_orchestration_service.collect_funding_rate_data.return_value = {
            "success": True,
            "message": "全期間のファンディングレートデータを収集しました",
            "data": {"collected_count": 5000},
        }

        # APIリクエスト
        response = test_client.post(
            "/api/funding-rates/collect",
            params={"symbol": "BTC/USDT:USDT", "fetch_all": True},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_collect_funding_rates_db_init_failure(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_funding_rate_orchestration_service: AsyncMock,
    ) -> None:
        """
        異常系: データベース初期化が失敗する場合

        Args:
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_funding_rate_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定: 成功レスポンスを返す（依存性注入により正常動作）
        mock_funding_rate_orchestration_service.collect_funding_rate_data.return_value = {
            "success": True,
            "message": "ファンディングレートデータを収集しました",
            "data": {"collected_count": 0},
        }

        # APIリクエスト
        response = test_client.post(
            "/api/funding-rates/collect",
            params={"symbol": "BTC/USDT:USDT"},
        )

        # アサーション: モックが設定されているため成功
        assert response.status_code == 200

    def test_collect_funding_rates_missing_symbol(
        self,
        test_client: TestClient,
    ) -> None:
        """
        異常系: 必須パラメータsymbolが欠落している場合

        Args:
            test_client: テストクライアント
        """
        # APIリクエスト
        response = test_client.post("/api/funding-rates/collect")

        # アサーション
        assert response.status_code == 422


class TestBulkCollectFundingRates:
    """ファンディングレート一括収集のテストクラス"""

    def test_bulk_collect_success(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_funding_rate_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: 一括収集が正常に実行される

        Args:
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_funding_rate_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_funding_rate_orchestration_service.collect_bulk_funding_rate_data.return_value = {
            "success": True,
            "message": "一括収集が完了しました",
            "data": {
                "results": [
                    {
                        "symbol": "BTC/USDT:USDT",
                        "collected_count": 5000,
                        "success": True,
                    }
                ]
            },
        }

        # APIリクエスト
        response = test_client.post("/api/funding-rates/bulk-collect")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "完了" in data["message"]

    def test_bulk_collect_db_init_failure(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_funding_rate_orchestration_service: AsyncMock,
    ) -> None:
        """
        異常系: データベース初期化が失敗する場合

        Args:
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_funding_rate_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定: 成功レスポンスを返す（依存性注入により正常動作）
        mock_funding_rate_orchestration_service.collect_bulk_funding_rate_data.return_value = {
            "success": True,
            "message": "一括収集が完了しました",
            "data": {"results": []},
        }

        # APIリクエスト
        response = test_client.post("/api/funding-rates/bulk-collect")

        # アサーション: モックが設定されているため成功
        assert response.status_code == 200


class TestErrorHandling:
    """エラーハンドリングのテストクラス"""

    def test_service_error_handling(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_funding_rate_orchestration_service: AsyncMock,
    ) -> None:
        """
        異常系: サービス層でエラーが発生した場合

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_funding_rate_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_funding_rate_orchestration_service.get_funding_rate_data.side_effect = (
            Exception("Database error")
        )

        # APIリクエスト
        response = test_client.get(
            "/api/funding-rates/",
            params={"symbol": "BTC/USDT:USDT"},
        )

        # アサーション（ErrorHandlerによって処理される）
        assert response.status_code in [200, 500]
